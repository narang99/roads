import cv2
import numpy as np
from PIL import Image
import torch
import functools
from enum import Enum
import matplotlib.pyplot as plt


class Label(Enum):
    BACKGROUND = 0
    ELEVATED_VEGETATION = 1


@functools.lru_cache(maxsize=1)
def get_cached_seg_former() -> "SegFormerElevatedVegetation":
    return SegFormerElevatedVegetation()


@functools.lru_cache(maxsize=40)
def cached_predict(img_path) -> np.ndarray:
    model = get_cached_seg_former()
    return model.predict(img_path)


class SegFormerElevatedVegetation:
    """
    example usage:
        model = SegFormerElevatedVegetation()
        pred = model.predict(<image-path>)
        road = model.get_mask(pred, CityScapesCls.ROAD)

        # simple numpy mask, you can overlay now
        orig = cv2.imread(<image-path>)
        # mark road as red, bgr
        orig[road] = [0, 0, 255]
        cv2.imshow(orig)
    """

    def __init__(self):
        # hardcoded for now
        from transformers import (
            SegformerImageProcessor,
            SegformerForSemanticSegmentation,
        )
        model_dir = "/Users/hariomnarang/Desktop/personal/roads/datasets/elevated-vegetation/checkpoint-3680"
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", do_reduce_labels=False
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            # num_labels=NUM_LABELS,
            # id2label=id2label,
            # label2id=label2id,
            # ignore_mismatched_sizes=True
        )

        self.processor = processor
        self.model = model

    def _predict(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        return pred_seg.numpy()

    def predict_bgr_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._predict(Image.fromarray(img))

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self._predict(img)

    def get_mask(pred, lbl: Label):
        return pred == lbl.value


def get_mask(pred, lbl: Label):
    return pred == lbl.value


def get_mask_with_labels(pred, lbls: list[Label]):
    mask = np.zeros_like(pred, dtype=bool)
    for lbl in lbls:
        mask |= pred == lbl.value
    return mask


def show_seg_mask(mask):
    label_by_name = {mem.value: mem.name for mem in Label}
    plt.figure(figsize=(5, 5))
    im = plt.imshow(mask, cmap="tab20", interpolation="nearest")
    plt.colorbar(im, ticks=np.unique(mask))
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.show()
    for k, v in label_by_name.items():
        print(f"{k:>20} -> {v}")
