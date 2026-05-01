import os
os.environ['JETSON_MODEL'] = 'JETSON_ORIN_NANO'

import atexit
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
SPEED          = 30   # base forward duty cycle (0–100); both wheels at this on a perfect straight
SPEED_ONE_BLOB = 20   # reduced base speed when only one boundary is visible (mid-turn)
SPEED_LOST     = 15   # speed while coasting through a brief detection dropout
BARRIER_SPEED  = 15   # pivot speed when a horizontal barrier is detected

# ── Steering controller knobs ─────────────────────────────────────────────────
# error ∈ [-1, +1]: -1 = lane center is at far left of frame, +1 = far right.
# left_duty  = SPEED + STEER_GAIN * error
# right_duty = SPEED - STEER_GAIN * error
# Higher gain = sharper turns + more wobble. Tune on the car.
STEER_GAIN          = 40
ERROR_ALPHA         = 0.25   # EWMA on raw error: smaller = smoother but laggier
TURN_SLOWDOWN       = 0.55   # at |err|=1, forward speed drops to SPEED * (1 - this).
                             # Lets the car physically rotate through tight turns instead of overshooting.
INSIDE_REVERSE_CAP  = 0      # disabled — inside wheel only slows, never reverses.
                             # Re-enable (e.g. 20) only if turns are too wide after barrier detection works.
BARRIER_WIDTH_FRAC  = 0.80   # single blob wider than this fraction of frame → horizontal barrier
LANE_WIDTH_DEFAULT  = 0.60   # initial lane width as a fraction of frame width
LANE_WIDTH_ALPHA    = 0.85   # EWMA on lane-width estimate (slow update)
SIDE_FLIP_PX        = 80     # in single-blob mode, the locked side flips only after the
                             # blob has crossed this far past camera center (hysteresis).
                             # Prevents both oscillation when blob is near center AND wrong
                             # locks during turns where the visible boundary genuinely moves
                             # across the frame.
LOST_FRAMES_HOLD    = 25     # ~2.5s at 10Hz: hold last command this long before stopping
STRIP_TOP_FRAC      = 0.50   # ignore top half of frame; only react to tape that is close to the car.

# ── Pin definitions (BCM numbering) ────────────────────────────────────────────
ENA = 17   # Left motor PWM   (board pin 11)
IN1 = 5    # Left  forward    (board pin 29)
IN2 = 6    # Left  backward   (board pin 31)
ENB = 27   # Right motor PWM  (board pin 13)
IN3 = 16   # Right forward    (board pin 36)
IN4 = 20   # Right backward   (board pin 38)

MIN_BLOB_AREA = 200 #for smaller blobs that are noise


# ── SoftPWM ────────────────────────────────────────────────────────────────────
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
_locked_side   = None    # "left"|"right"|None — sticky identity for single-blob mode;
                         # set on first single-blob frame, cleared when both blobs reappear
_barrier_turning   = False  # True while executing a barrier pivot manoeuvre
_barrier_direction = None   # "left"|"right" — which way to pivot during barrier


def reset_steering_state():
    """Wipe per-run state so a stale lock or ewma can't carry into a new attempt.
    Lane-width estimate is intentionally kept — it's a slowly-changing physical
    property and a good estimate from the previous run beats the default."""
    global _prev_left_x, _prev_right_x, _error_ewma, _lost_count, _locked_side
    global _barrier_turning, _barrier_direction
    _prev_left_x       = None
    _prev_right_x      = None
    _error_ewma        = 0.0
    _lost_count        = 0
    _locked_side       = None
    _barrier_turning   = False
    _barrier_direction = None


def decide_steering(mask):
    """
    Returns (error, regime, debug).
      error  ∈ [-1, +1]   — smoothed; >0 means lane center is right of camera center
      regime ∈ {"two", "one_left", "one_right", "lost"}
      debug  — dict for the on-screen overlay (lane_x, left_x, right_x, lane_width)
    """
    global _prev_left_x, _prev_right_x, _lane_width_px, _error_ewma, _lost_count, _locked_side

    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    if _lane_width_px is None:
        _lane_width_px = LANE_WIDTH_DEFAULT * w

    # Search strip: skip the top horizon area; everything below is fair game.
    # Camera is tilted down hard enough that "near tape" actually appears in the
    # middle of the frame, not the bottom.
    strip_start = int(h * STRIP_TOP_FRAC)
    bottom = mask_u8[strip_start:, :]

    # Light morphological open to suppress speckle before connected components.
    kernel = np.ones((3, 3), np.uint8)
    bottom = cv2.morphologyEx(bottom, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(bottom)
    blobs = [
        (float(centroids[i][0]),            # [0] cx  (strip coords)
         int(stats[i, cv2.CC_STAT_AREA]),   # [1] area
         int(stats[i, cv2.CC_STAT_WIDTH]),  # [2] blob width
         int(stats[i, cv2.CC_STAT_LEFT]),   # [3] left edge X
         int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) + strip_start)
        # [4] bottom edge Y in original-frame coordinates
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA
    ]

    debug = {"left_x": None, "right_x": None, "lane_x": None,
             "lane_width": _lane_width_px, "n_blobs": len(blobs),
             "barrier_dir": None}

    # ── Regime: lost ──────────────────────────────────────────────────────────
    if not blobs:
        _lost_count += 1
        if _lost_count > LOST_FRAMES_HOLD:
            _prev_left_x  = None
            _prev_right_x = None
            _locked_side  = None
            _error_ewma   = 0.0
        return _error_ewma, "lost", debug

    # ── Regime: barrier — single connected blob spanning 80%+ of frame width ─
    # A 90° corner tape appears as one wide connected blob. Two separate lane
    # boundaries will never individually be this wide.
    for cx, area, blob_w, blob_left, bottom_y in blobs:
        if blob_w > BARRIER_WIDTH_FRAC * w:
            right_gap = w - (blob_left + blob_w)
            left_gap  = blob_left
            debug["barrier_dir"] = "right" if right_gap >= left_gap else "left"
            debug["lane_x"] = w / 2
            return 0.0, "barrier", debug

    _lost_count = 0
    blobs_sorted = sorted(blobs, key=lambda b: b[0])  # sort by centroid X

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
        _locked_side = None   # both visible — release any single-blob lock

        # Proximity override — only for tape in the bottom 15% of the frame
        # (physically close to the car). A distant barrier in the upper part of
        # the detection strip must NOT trigger this.
        left_bottom_y  = blobs_sorted[0][4]
        right_bottom_y = blobs_sorted[-1][4]
        if left_x > w * 0.45 and left_bottom_y > h * 0.85:
            debug["barrier_dir"] = "right"
            debug["lane_x"] = lane_x
            return 0.0, "barrier", debug
        if right_x < w * 0.55 and right_bottom_y > h * 0.85:
            debug["barrier_dir"] = "left"
            debug["lane_x"] = lane_x
            return 0.0, "barrier", debug

    # ── Regime: one blob — sticky-with-hysteresis identity ───────────────────
    else:
        cx = blobs_sorted[0][0]
        center = w / 2

        # Initial lock: prefer proximity to last 2-blob frame, fall back to
        # which half of the frame the blob is in.
        if _locked_side is None:
            d_left  = abs(cx - _prev_left_x)  if _prev_left_x  is not None else float("inf")
            d_right = abs(cx - _prev_right_x) if _prev_right_x is not None else float("inf")
            if d_left == float("inf") and d_right == float("inf"):
                _locked_side = "left" if cx < center else "right"
            elif d_left == float("inf"):
                _locked_side = "right"
            elif d_right == float("inf"):
                _locked_side = "left"
            else:
                _locked_side = "left" if d_left <= d_right else "right"
        else:
            # Already locked. Allow it to flip only when the blob has clearly
            # crossed past camera center to the other side (SIDE_FLIP_PX of
            # hysteresis prevents flapping when the blob hovers near center).
            if _locked_side == "left" and cx > center + SIDE_FLIP_PX:
                _locked_side = "right"
            elif _locked_side == "right" and cx < center - SIDE_FLIP_PX:
                _locked_side = "left"

        if _locked_side == "left":
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
    global annotated_frame, last_inference_ts, _barrier_turning, _barrier_direction
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

            left_duty = right_duty = 0
            use_normal_steering = False

            if not is_autonomous:
                stop_motors()
                _barrier_turning   = False
                _barrier_direction = None
            elif _barrier_turning:
                # Mid-pivot: keep turning until we see both lane lines again.
                if regime == "two":
                    _barrier_turning   = False
                    _barrier_direction = None
                    use_normal_steering = True
                else:
                    if _barrier_direction == "right":
                        left_duty, right_duty = BARRIER_SPEED, -BARRIER_SPEED
                    else:
                        left_duty, right_duty = -BARRIER_SPEED, BARRIER_SPEED
                    drive(left_duty, right_duty)
            elif regime == "barrier":
                _barrier_turning   = True
                _barrier_direction = dbg.get("barrier_dir") or "right"
                if _barrier_direction == "right":
                    left_duty, right_duty = BARRIER_SPEED, -BARRIER_SPEED
                else:
                    left_duty, right_duty = -BARRIER_SPEED, BARRIER_SPEED
                drive(left_duty, right_duty)
            elif regime == "lost":
                # Hold last command at reduced speed for a short window, then stop.
                if _lost_count <= LOST_FRAMES_HOLD:
                    left_duty  = max(0, min(100, SPEED_LOST + STEER_GAIN * error))
                    right_duty = max(0, min(100, SPEED_LOST - STEER_GAIN * error))
                drive(left_duty, right_duty)
            else:
                use_normal_steering = True

            if use_normal_steering:
                base_speed = SPEED_ONE_BLOB if regime in ("one_left", "one_right") else SPEED
                slow_factor = 1.0 - TURN_SLOWDOWN * abs(error)
                base = base_speed * slow_factor
                left_duty  = max(-INSIDE_REVERSE_CAP, min(100, base + STEER_GAIN * error))
                right_duty = max(-INSIDE_REVERSE_CAP, min(100, base - STEER_GAIN * error))
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
            auto_str  = "AUTO" if is_autonomous else "IDLE"
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
        try:
            key = input().strip().lower()
        except EOFError:
            # SSH disconnected or stdin closed — clean up properly
            shutdown(None, None)
            return
        if key == 'q':
            shutdown(None, None)
        else:
            with auto_lock:
                autonomous = not autonomous
                state = autonomous
            if state:
                reset_steering_state()
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
        small = cv2.resize(frame, (640, 480))
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(1 / 15)   # cap stream at 15 fps

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
    if state:
        reset_steering_state()
    else:
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
atexit.register(lambda: GPIO.cleanup())


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
