# code to generate segmentation using cityscapes pretrained modelsplt

import cv2
import numpy as np
from PIL import Image
import torch
import functools
from enum import Enum
import matplotlib.pyplot as plt

class CityScapesCls(Enum):
    ROAD = 0
    SIDEWALK = 1
    BUILDING = 2
    WALL = 3
    FENCE = 4
    POLE = 5
    TRAFFIC_LIGHT = 6
    TRAFFIC_SIGN = 7
    VEGETATION = 8
    TERRAIN = 9
    SKY = 10
    PERSON = 11
    RIDER = 12
    CAR = 13
    TRUCK = 14
    BUS = 15
    TRAIN = 16
    MOTORCYCLE = 17
    BICYCLE = 18


@functools.lru_cache(maxsize=1)
def get_cached_seg_former() -> "SegFormerCityScapes":
    return SegFormerCityScapes()

@functools.lru_cache(maxsize=40)
def cached_predict(img_path) -> np.ndarray:
    model = get_cached_seg_former()
    return model.predict(img_path)

class SegFormerCityScapes:
    """
    example usage:
        model = SegFormerCityScapes()
        pred = model.predict(<image-path>)
        road = model.get_mask(pred, CityScapesCls.ROAD)

        # simple numpy mask, you can overlay now
        orig = cv2.imread(<image-path>)
        # mark road as red, bgr
        orig[road] = [0, 0, 255]
        cv2.imshow(orig)
    """
    def __init__(self):
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        feat = SegformerFeatureExtractor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model = model
        self.feat = feat

    def _predict(self, pil_image):
        img = pil_image
        orig = np.array(img)
        h, w, _ = orig.shape

        inputs = self.feat(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return torch.argmax(logits, dim=1)[0].cpu().numpy()
    
    def predict_bgr_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._predict(Image.fromarray(img))

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self._predict(img)

    def get_mask(pred, lbl: CityScapesCls):
        return (pred == lbl.value)

def get_mask(pred, lbl: CityScapesCls):
    return (pred == lbl.value)


def get_mask_with_labels(pred, lbls: list[CityScapesCls]):
    mask = np.zeros_like(pred, dtype=bool)
    for lbl in lbls:
        mask |= (pred == lbl.value)
    return mask


def show_seg_mask(mask):
    label_by_name = {mem.value: mem.name for mem in CityScapesCls}
    plt.figure(figsize=(5, 5))
    im = plt.imshow(mask, cmap="tab20", interpolation="nearest")
    plt.colorbar(im, ticks=np.unique(mask))
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.show()
    for k,v in label_by_name.items():
        print(f"{k:>20} -> {v}")
