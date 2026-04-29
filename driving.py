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
SPEED          = 50   # base forward duty cycle (0–100); both wheels at this on a perfect straight
SPEED_LOST     = 25   # speed while coasting through a brief detection dropout

# ── Steering controller knobs ─────────────────────────────────────────────────
# error ∈ [-1, +1]: -1 = lane center is at far left of frame, +1 = far right.
# left_duty  = SPEED + STEER_GAIN * error
# right_duty = SPEED - STEER_GAIN * error
# Higher gain = sharper turns + more wobble. Tune on the car.
STEER_GAIN          = 35
ERROR_ALPHA         = 0.65   # EWMA on raw error: smaller = smoother but laggier
LANE_WIDTH_DEFAULT  = 0.60   # initial lane width as a fraction of frame width
LANE_WIDTH_ALPHA    = 0.85   # EWMA on lane-width estimate (slow update)
BOUNDARY_MATCH_PX   = 120    # max horizontal jump for matching a blob to last frame's left/right boundary
LOST_FRAMES_HOLD    = 15     # ~1.5s at 10Hz: hold last command this long before stopping

# ── Pin definitions (BCM numbering) ────────────────────────────────────────────
ENA = 17   # Left motor PWM   (board pin 11)
IN1 = 5    # Left  forward    (board pin 29)
IN2 = 6    # Left  backward   (board pin 31)
ENB = 27   # Right motor PWM  (board pin 13)
IN3 = 16   # Right forward    (board pin 36)
IN4 = 20   # Right backward   (board pin 38)

MIN_BLOB_AREA = 200 #for smaller blobs that are noise


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
# drive() is the primitive: each side takes a signed duty cycle in [-100, 100].
# Positive = forward, negative = backward. Everything else is a wrapper.
def drive(left, right):
    left  = max(-100, min(100, left))
    right = max(-100, min(100, right))
    if left >= 0:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    else:
        GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    if right >= 0:
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    else:
        GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(abs(left))
    pwm_right.ChangeDutyCycle(abs(right))

def stop_motors():
    drive(0, 0)
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.LOW)

def forward(speed=SPEED):
    drive(speed, speed)

def backward(speed=SPEED):
    drive(-speed, -speed)

# Pivot helpers — kept for manual testing only; the autonomous loop uses drive()
# directly via differential speed control.
def turn_left(speed=SPEED):
    drive(-speed, speed)

def turn_right(speed=SPEED):
    drive(speed, -speed)


# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    for _idx in [1, 2]:
        cap = cv2.VideoCapture(_idx, cv2.CAP_V4L2)
        if cap.isOpened():
            break

if not cap.isOpened():
    print("ERROR: could not open camera at /dev/video0, /dev/video1, or /dev/video2")
    sys.exit(1)

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

# Watchdog: inference_loop bumps this on every successful pass.
# If it goes stale while autonomous is on, the watchdog stops the motors
# so a hung capture or GPU stall can't leave the wheels spinning.
last_inference_ts = time.monotonic()
WATCHDOG_TIMEOUT  = 1.0   # seconds


# ── Thread 1: capture ──────────────────────────────────────────────────────────
# Runs as fast as the camera allows; decouples capture rate from inference rate.
def capture_loop():
    global latest_frame
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with frame_lock:
                latest_frame = frame
        except Exception as e:
            print(f"capture_loop error: {e}")
            time.sleep(0.1)


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


# ── Steering controller state ─────────────────────────────────────────────────
# Persisted across frames so we can track which boundary is which when only one
# is visible, smooth the error signal, and hold last command through dropouts.
_prev_left_x   = None    # last known x of left  boundary (in mask coords)
_prev_right_x  = None    # last known x of right boundary (in mask coords)
_lane_width_px = None    # running estimate of lane width in pixels (mask coords)
_error_ewma    = 0.0     # smoothed steering error in [-1, +1]
_lost_count    = 0       # consecutive frames with no usable detection


def decide_steering(mask):
    """
    Returns (error, regime, debug).
      error  ∈ [-1, +1]   — smoothed; >0 means lane center is right of camera center
      regime ∈ {"two", "one_left", "one_right", "lost"}
      debug  — dict for the on-screen overlay (lane_x, left_x, right_x, lane_width)
    """
    global _prev_left_x, _prev_right_x, _lane_width_px, _error_ewma, _lost_count

    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    if _lane_width_px is None:
        _lane_width_px = LANE_WIDTH_DEFAULT * w

    # Closest-to-car strip. Bottom third is fine; bigger strip = more context.
    bottom = mask_u8[h * 2 // 3:, :]

    # Light morphological open to suppress speckle before connected components.
    kernel = np.ones((3, 3), np.uint8)
    bottom = cv2.morphologyEx(bottom, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bottom)
    blobs = [
        (float(centroids[i][0]), int(stats[i, cv2.CC_STAT_AREA]))
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA
    ]

    debug = {"left_x": None, "right_x": None, "lane_x": None,
             "lane_width": _lane_width_px, "n_blobs": len(blobs)}

    # ── Regime: lost ──────────────────────────────────────────────────────────
    if not blobs:
        _lost_count += 1
        return _error_ewma, "lost", debug

    _lost_count = 0
    blobs_sorted = sorted(blobs, key=lambda b: b[0])

    # ── Regime: two or more blobs — use outermost as the two boundaries ──────
    if len(blobs_sorted) >= 2:
        left_x  = blobs_sorted[0][0]
        right_x = blobs_sorted[-1][0]
        observed_width = right_x - left_x
        # Only update lane width from sane observations (avoid degenerate
        # near-zero widths from two blobs of the same boundary fragmented).
        if observed_width > 0.2 * w:
            _lane_width_px = (LANE_WIDTH_ALPHA * _lane_width_px
                              + (1 - LANE_WIDTH_ALPHA) * observed_width)
        _prev_left_x, _prev_right_x = left_x, right_x
        lane_x = (left_x + right_x) / 2
        regime = "two"

    # ── Regime: one blob — assign to a side using history ─────────────────────
    else:
        cx = blobs_sorted[0][0]

        # Distance to last known left/right; pick the closer match.
        d_left  = abs(cx - _prev_left_x)  if _prev_left_x  is not None else float("inf")
        d_right = abs(cx - _prev_right_x) if _prev_right_x is not None else float("inf")

        if d_left == float("inf") and d_right == float("inf"):
            # No history yet — fall back to "left half = left boundary"
            is_left = cx < w / 2
        elif d_left <= BOUNDARY_MATCH_PX or d_right <= BOUNDARY_MATCH_PX:
            is_left = d_left <= d_right
        else:
            # Neither last position matches → blob jumped. Treat as the side
            # we last saw anchored. If we have both, pick by raw position.
            is_left = cx < w / 2

        if is_left:
            _prev_left_x = cx
            lane_x = cx + _lane_width_px / 2
            regime = "one_left"
        else:
            _prev_right_x = cx
            lane_x = cx - _lane_width_px / 2
            regime = "one_right"

    # ── Convert to error and smooth ────────────────────────────────────────────
    raw_error = (lane_x - w / 2) / (w / 2)
    raw_error = max(-1.0, min(1.0, raw_error))
    _error_ewma = ERROR_ALPHA * _error_ewma + (1 - ERROR_ALPHA) * raw_error

    debug["lane_x"]   = lane_x
    debug["left_x"]   = _prev_left_x
    debug["right_x"]  = _prev_right_x
    debug["lane_width"] = _lane_width_px
    return _error_ewma, regime, debug

# ── Thread 2: inference + motor control ────────────────────────────────────────
# Reads the latest camera frame, runs the segmentation model, decides a steering
# command, drives the motors, then writes an annotated frame for the web stream.
def inference_loop():
    global annotated_frame, last_inference_ts
    while True:
        try:
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
            error, regime, dbg = decide_steering(pred_mask)

            with auto_lock:
                is_autonomous = autonomous

            if not is_autonomous:
                left_duty = right_duty = 0
                stop_motors()
            elif regime == "lost":
                # Hold last command at reduced speed for a short window, then stop.
                if _lost_count <= LOST_FRAMES_HOLD:
                    left_duty  = max(0, min(100, SPEED_LOST + STEER_GAIN * error))
                    right_duty = max(0, min(100, SPEED_LOST - STEER_GAIN * error))
                else:
                    left_duty = right_duty = 0
                drive(left_duty, right_duty)
            else:
                # Normal proportional control. Clamp to [0, 100] so we never
                # auto-reverse a wheel — keeps motion smooth for the demo.
                left_duty  = max(0, min(100, SPEED + STEER_GAIN * error))
                right_duty = max(0, min(100, SPEED - STEER_GAIN * error))
                drive(left_duty, right_duty)

            # --- Annotate frame for streaming ---
            vis = overlay_mask(frame, pred_mask.astype(int))
            # Draw lane center marker if available (in mask coords → frame coords)
            if dbg["lane_x"] is not None:
                fx = int(dbg["lane_x"] * frame.shape[1] / pred_mask.shape[1])
                fy = int(frame.shape[0] * 5 / 6)
                cv2.circle(vis, (fx, fy), 8, (0, 255, 255), -1)
                cv2.line(vis, (frame.shape[1] // 2, fy - 20),
                              (frame.shape[1] // 2, fy + 20), (255, 255, 255), 1)
            auto_str  = "AUTO" if is_autonomous else "IDLE (press Enter)"
            label1 = (f"{auto_str}  regime:{regime}  err:{error:+.2f}"
                      f"  L:{int(left_duty)} R:{int(right_duty)}")
            label2 = (f"blobs:{dbg['n_blobs']}  lane_w:{int(dbg['lane_width'])}"
                      f"  lost:{_lost_count}  conf:{max_conf:.2f}")
            cv2.putText(vis, label1, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, label2, (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            with annotated_lock:
                annotated_frame = vis

            last_inference_ts = time.monotonic()
            time.sleep(0.1)   # ~10 Hz — reduces GPU/CPU load and battery drain
        except Exception as e:
            print(f"inference_loop error: {e}")
            stop_motors()
            time.sleep(0.1)


# ── Thread 3: watchdog ─────────────────────────────────────────────────────────
# If inference_loop hangs (camera stall, GPU lockup, etc.) the wheels would keep
# spinning at whatever the last command was. This thread cuts power if no
# successful inference has happened in WATCHDOG_TIMEOUT seconds.
def watchdog_loop():
    while True:
        time.sleep(WATCHDOG_TIMEOUT / 2)
        with auto_lock:
            is_autonomous = autonomous
        if is_autonomous and (time.monotonic() - last_inference_ts) > WATCHDOG_TIMEOUT:
            print("WATCHDOG: inference stalled, stopping motors")
            stop_motors()


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
_shutdown_started = False
_shutdown_lock    = threading.Lock()

def shutdown(sig, frame):
    global _shutdown_started
    with _shutdown_lock:
        if _shutdown_started:
            return
        _shutdown_started = True
    print("\nShutting down …")
    try:
        stop_motors()
        pwm_left.stop()
        pwm_right.stop()
        GPIO.cleanup()
        cap.release()
    except Exception as e:
        print(f"shutdown error: {e}")
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
    threading.Thread(target=watchdog_loop,  daemon=True).start()
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False),
                     daemon=True).start()

    print(f"Open http://<jetson-ip>:{PORT} in your browser")
    keyboard_listener()   # runs on main thread — stdin works correctly
