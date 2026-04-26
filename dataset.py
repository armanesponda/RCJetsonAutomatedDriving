import os
import json
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Roboflow COCO Segmentation export layout:
#   data/
#     train/  images/ + _annotations.coco.json
#     valid/  images/ + _annotations.coco.json
#     test/   images/ + _annotations.coco.json

class LaneDataset(Dataset):
    def __init__(self, split_dir, size=(520, 520), augment=False):
        self.img_dir = os.path.join(split_dir, "images")
        self.size    = size
        self.augment = augment

        with open(os.path.join(split_dir, "_annotations.coco.json")) as f:
            coco = json.load(f)

        self.images = coco["images"]

        # image_id → list of annotations
        self.anns = {}
        for ann in coco["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info  = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        w, h  = image.size

        # Rasterize polygon annotations → binary mask (0=background, 1=lane)
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for ann in self.anns.get(info["id"], []):
            for seg in ann["segmentation"]:
                if len(seg) >= 6:
                    poly = [(float(seg[i]), float(seg[i + 1])) for i in range(0, len(seg), 2)]
                    draw.polygon(poly, fill=1)

        image = image.resize(self.size, Image.BILINEAR)
        mask  = mask.resize(self.size,  Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        mask  = torch.as_tensor(list(mask.getdata()), dtype=torch.long)
        mask  = mask.view(self.size[1], self.size[0])
        return image, mask
