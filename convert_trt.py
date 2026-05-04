"""
Run this ONCE on the Jetson to convert best_model.pth → best_model_trt.pth.
After that, driving.py picks up the TRT engine automatically.

    python3 convert_trt.py

Takes a few minutes the first time. Subsequent runs of driving.py load in seconds.
"""
import os
os.environ['JETSON_MODEL'] = 'JETSON_ORIN_NANO'

import torch
from model import build_model

MODEL_PATH = "best_model.pth"
TRT_PATH   = "best_model_trt.pth"
# (width, height) → tensor is (1, 3, height, width)
INPUT_H, INPUT_W = 360, 640


class _Wrapper(torch.nn.Module):
    """Strip the DeepLabV3 dict output so torch2trt sees a plain tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def main():
    try:
        from torch2trt import torch2trt
    except ImportError:
        print("torch2trt not found. Install it from: https://github.com/NVIDIA-AI-IOT/torch2trt")
        return

    print(f"Loading checkpoint from {MODEL_PATH} ...")
    base = build_model(pretrained=False)
    base.load_state_dict(torch.load(MODEL_PATH, map_location="cuda"), strict=False)
    base = base.cuda().eval()

    wrapped = _Wrapper(base).cuda().eval()

    dummy = torch.ones(1, 3, INPUT_H, INPUT_W).cuda()

    print("Converting to TensorRT FP16 — this takes a few minutes ...")
    trt_model = torch2trt(
        wrapped,
        [dummy],
        fp16_mode=True,
        max_workspace_size=1 << 26,   # 64 MB workspace
    )

    torch.save(trt_model.state_dict(), TRT_PATH)
    print(f"Done. TRT engine saved to {TRT_PATH}")

    # Quick sanity check — compare outputs
    with torch.no_grad():
        out_pt  = wrapped(dummy)
        out_trt = trt_model(dummy)
    diff = (out_pt - out_trt).abs().max().item()
    print(f"Max output difference PyTorch vs TRT: {diff:.5f}  (should be < 0.01)")


if __name__ == "__main__":
    main()
