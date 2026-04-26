import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from flask import Flask, Response
from model import load_checkpoint

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
INPUT_SIZE = (640, 360)
PORT       = 5001

# ── Camera ─────────────────────────────────────────────────────────────
# USB webcam:
cap = cv2.VideoCapture(0)

# CSI camera (swap in if you're using the ribbon-cable cam instead):
# GST = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,"
#        "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! "
#        "videoconvert ! video/x-raw,format=BGR ! appsink")
# cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)

# ── Model ───────────────────────────────────────────────────────────────
print(f"Loading {MODEL_PATH} on {DEVICE} ...")
model = load_checkpoint(MODEL_PATH, device=DEVICE)
model.eval()
print("Model ready.")

app = Flask(__name__)


def preprocess(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img    = Image.fromarray(rgb).resize(INPUT_SIZE, Image.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)          # [1, 3, H, W]


def overlay_mask(frame, mask):
    """Blend a green tint over predicted lane pixels."""
    h, w        = frame.shape[:2]
    mask_full   = cv2.resize(mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    green_layer = np.zeros_like(frame)
    green_layer[mask_full == 1] = (0, 255, 0)
    return cv2.addWeighted(frame, 0.7, green_layer, 0.3, 0)


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        tensor = preprocess(frame).to(DEVICE)
        with torch.no_grad():
            out  = model(tensor)["out"]          # [1, 2, H, W]
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()   # [H, W]

        annotated = overlay_mask(frame, pred)

        # Burn device + class info into the corner so you can confirm GPU is active
        label = f"{DEVICE.upper()}  lane px: {int(pred.sum())}"
        cv2.putText(annotated, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buf = cv2.imencode(".jpg", annotated)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return (
        "<html><body style='background:#111;margin:0'>"
        "<img src='/video' style='width:100%;height:auto'>"
        "</body></html>"
    )


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print(f"Open http://<jetson-ip>:{PORT} in your browser")
    app.run(host="0.0.0.0", port=PORT, debug=False)
