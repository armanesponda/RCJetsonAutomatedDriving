import os
os.environ['JETSON_MODEL'] = 'JETSON_ORIN_NANO'

import signal
import sys
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import Jetson.GPIO as GPIO
from flask import Flask, Response, render_template_string

from model import load_checkpoint

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
INPUT_SIZE = (640, 360)   # width, height fed to the model
PORT       = 5000
SPEED          = 50   # straight speed (0–100)
SPEED_TURN     = 35   # speed while turning
SPEED_RECOVER  = 30   # speed while searching for lane (no blobs)

# ── Pin definitions (BCM numbering) ────────────────────────────────────────────
ENA = 17   # Left motor PWM   (board pin 11)
IN1 = 5    # Left  forward    (board pin 29)
IN2 = 6    # Left  backward   (board pin 31)
ENB = 27   # Right motor PWM  (board pin 13)
IN3 = 16   # Right forward    (board pin 36)
IN4 = 20   # Right backward   (board pin 38)

MIN_BLOB_AREA = 200 #for smaller blobs that are noise
STEER_DEADBAND = 0.10 #10% of the frame width counts as centered


# ── SoftPWM ────────────────────────────────────────────────────────────────────
# The Jetson Orin Nano does not expose hardware PWM on the same pins as a
# Raspberry Pi, so we replicate it in software: a background thread toggles
# the enable pin at the requested duty cycle.
class SoftPWM:
    def __init__(self, pin, freq=100):
        self.pin    = pin
        self.period = 1.0 / freq
        self.duty   = 0
        self._running = False
        self._thread  = None

    def start(self, duty):
        self.duty     = duty
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            dc = self.duty
            if dc <= 0:
                GPIO.output(self.pin, GPIO.LOW)
                time.sleep(self.period)
            elif dc >= 100:
                GPIO.output(self.pin, GPIO.HIGH)
                time.sleep(self.period)
            else:
                GPIO.output(self.pin, GPIO.HIGH)
                time.sleep(self.period * dc / 100.0)
                GPIO.output(self.pin, GPIO.LOW)
                time.sleep(self.period * (100.0 - dc) / 100.0)

    def ChangeDutyCycle(self, duty):
        self.duty = duty

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.2)
        GPIO.output(self.pin, GPIO.LOW)


# ── GPIO & PWM setup ───────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)
pwm_left  = SoftPWM(ENA, freq=100)
pwm_right = SoftPWM(ENB, freq=100)
pwm_left.start(0)
pwm_right.start(0)


# ── Motor commands ─────────────────────────────────────────────────────────────
def stop_motors():
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

def forward(speed=SPEED):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def backward(speed=SPEED):
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def turn_left(speed=SPEED):
    # Left wheel reverses while right wheel goes forward → pivot left
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def turn_right(speed=SPEED):
    # Right wheel reverses while left wheel goes forward → pivot right
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)


# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    for _idx in [1, 2]:
        cap = cv2.VideoCapture(_idx, cv2.CAP_V4L2)
        if cap.isOpened():
            break

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Shared state between threads — always acquire the matching lock before reading/writing.
latest_frame    = None   # raw BGR frame from camera
annotated_frame = None   # frame with lane mask overlay drawn on it
frame_lock      = threading.Lock()
annotated_lock  = threading.Lock()

autonomous      = False  # toggled by Enter key; car only drives when True
auto_lock       = threading.Lock()


# ── Thread 1: capture ──────────────────────────────────────────────────────────
# Runs as fast as the camera allows; decouples capture rate from inference rate.
def capture_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame


# ── Preprocessing & visualization helpers ─────────────────────────────────────
def preprocess(frame):
    """Convert a BGR camera frame to a normalised [1,3,H,W] tensor."""
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img    = Image.fromarray(rgb).resize(INPUT_SIZE, Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor,
                          mean=[0.485, 0.456, 0.406],
                          std =[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)   # [1, 3, H, W]


def overlay_mask(frame, mask):
    """Blend a green tint over every pixel the model classifies as lane."""
    h, w = frame.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    green_layer = np.zeros_like(frame)
    green_layer[mask_resized == 1] = (0, 255, 0)
    return cv2.addWeighted(frame, 0.7, green_layer, 0.3, 0)


def decide_steering(mask):
    """Returns (command, speed). 'recover' means no lane visible — turn right to search."""
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape

    # Only use the bottom third — closest tape to the car
    bottom = mask_u8[h * 2 // 3:, :]

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bottom)
    blobs = [
        (centroids[i][0], stats[i, cv2.CC_STAT_AREA])
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA
    ]

    if len(blobs) == 0:
        return "recover", SPEED_RECOVER

    if len(blobs) == 1:
        # Only one boundary visible — steer away from it
        cx = blobs[0][0]
        return ("turn_right" if cx < w / 2 else "turn_left"), SPEED_TURN

    # Two or more blobs — find leftmost and rightmost boundary centroids
    blobs_sorted = sorted(blobs, key=lambda b: b[0])
    cx_mid = (blobs_sorted[0][0] + blobs_sorted[-1][0]) / 2

    # Steer toward the midpoint (lane center)
    error = (cx_mid - w / 2) / (w / 2)

    if abs(error) < STEER_DEADBAND:
        return "forward", SPEED
    elif error < 0:
        return "turn_left", SPEED_TURN
    else:
        return "turn_right", SPEED_TURN

# ── Thread 2: inference + motor control ────────────────────────────────────────
# Reads the latest camera frame, runs the segmentation model, decides a steering
# command, drives the motors, then writes an annotated frame for the web stream.
def inference_loop():
    global annotated_frame
    while True:
        # Grab the most recent frame (non-blocking — skip if nothing new yet)
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        # --- Inference ---
        tensor = preprocess(frame).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)["out"]          # [1, 2, H, W]  (logits)

        probs     = F.softmax(out, dim=1)       # convert logits → probabilities
        lane_prob = probs[0, 1]                 # [H, W]  probability of lane class
        max_conf  = float(lane_prob.max().cpu())
        pred_mask = (lane_prob.cpu().numpy() > 0.25)   # boolean [H, W]

        # --- Motor control ---
        command, speed = decide_steering(pred_mask)

        with auto_lock:
            is_autonomous = autonomous

        if not is_autonomous:
            stop_motors()
        elif command == "recover":
            turn_right(SPEED_RECOVER)   # pivot right until lane reappears
        else:
            {"forward": forward, "turn_left": turn_left, "turn_right": turn_right}[command](speed)

        # --- Annotate frame for streaming ---
        vis = overlay_mask(frame, pred_mask.astype(int))
        display_cmd = "recover→right" if command == "recover" else command
        display_spd = speed if is_autonomous else 0
        auto_str    = "AUTO" if is_autonomous else "IDLE (press Enter)"
        label = (f"{auto_str}  cmd:{display_cmd}  spd:{display_spd}"
                 f"  lane px:{int(pred_mask.sum())}  conf:{max_conf:.2f}")
        cv2.putText(vis, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with annotated_lock:
            annotated_frame = vis

        time.sleep(0.1)   # ~10 Hz — reduces GPU/CPU load and battery drain


# ── Flask: MJPEG stream ────────────────────────────────────────────────────────
app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RC Autonomous Drive</title>
  <style>
    body { background:#111; margin:0; display:flex; flex-direction:column;
           align-items:center; color:#eee; font-family:sans-serif; padding:12px; }
    img  { width:100%; max-width:800px; border-radius:6px; }
    p    { font-size:13px; color:#888; margin-top:8px; }
    #btn { margin:14px 0; padding:16px 0; font-size:24px; font-weight:bold;
           border:none; border-radius:10px; cursor:pointer; width:260px; }
    #btn.idle    { background:#27ae60; color:#fff; }
    #btn.running { background:#e74c3c; color:#fff; }
  </style>
</head>
<body>
  <h2>RC Autonomous Drive</h2>
  <img src="/stream" alt="camera feed">
  <button id="btn" class="idle">START</button>
  <p>Green overlay = detected lane &nbsp;|&nbsp; cmd shown top-left</p>
<script>
const btn = document.getElementById('btn');
function refresh() {
  fetch('/status').then(r => r.json()).then(d => {
    btn.textContent = d.autonomous ? 'STOP' : 'START';
    btn.className   = d.autonomous ? 'running' : 'idle';
  });
}
btn.addEventListener('click', () =>
  fetch('/toggle', {method:'POST'}).then(refresh)
);
setInterval(refresh, 1000);
</script>
</body>
</html>
"""

def keyboard_listener():
    global autonomous
    print("Controls: Enter = start/stop autonomous mode | q+Enter = quit")
    while True:
        key = input().strip().lower()
        if key == 'q':
            shutdown(None, None)
        else:
            with auto_lock:
                autonomous = not autonomous
                state = autonomous
            if state:
                print("Autonomous ON  — car will drive")
            else:
                stop_motors()
                print("Autonomous OFF — motors stopped")

def mjpeg_generator():
    while True:
        with annotated_lock:
            frame = annotated_frame
        if frame is None:
            # Model not ready yet — send the raw camera frame instead
            with frame_lock:
                frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        small = cv2.resize(frame, (320, 240))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(1 / 10)   # cap stream at 10 fps to save bandwidth

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    from flask import jsonify
    with auto_lock:
        state = autonomous
    return jsonify(autonomous=state)

@app.route("/toggle", methods=["POST"])
def toggle():
    from flask import jsonify
    global autonomous
    with auto_lock:
        autonomous = not autonomous
        state = autonomous
    if not state:
        stop_motors()
    return jsonify(autonomous=state)


# ── Clean shutdown ─────────────────────────────────────────────────────────────
def shutdown(sig, frame):
    print("\nShutting down …")
    stop_motors()
    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGHUP,  shutdown)
signal.signal(signal.SIGINT,  shutdown)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {MODEL_PATH} on {DEVICE} …")
    model = load_checkpoint(MODEL_PATH, device=DEVICE)
    model.eval()
    print("Model ready.")

    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False),
                     daemon=True).start()

    print(f"Open http://<jetson-ip>:{PORT} in your browser")
    keyboard_listener()   # runs on main thread — stdin works correctly
