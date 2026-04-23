import torch
import torch.nn as nn
import torchvision

NUM_CLASSES = 2  # 0 = background, 1 = lane (blue tape)

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=weights
    )
    # Replace final conv to match our number of classes (pretrained head outputs 21)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def load_checkpoint(path, device="cpu"):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
