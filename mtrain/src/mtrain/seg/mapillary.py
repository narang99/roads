# code to generate segmentation using cityscapes pretrained modelsplt

import cv2
import numpy as np
from PIL import Image
import torch
import functools
from enum import Enum
import matplotlib.pyplot as plt
from .draw import show_segmentation_pred



class Label(Enum):
    BIRD = 0
    GROUND_ANIMAL = 1
    CURB = 2
    FENCE = 3
    GUARD_RAIL = 4
    BARRIER = 5
    WALL = 6
    BIKE_LANE = 7
    CROSSWALK_PLAIN = 8
    CURB_CUT = 9
    PARKING = 10
    PEDESTRIAN_AREA = 11
    RAIL_TRACK = 12
    ROAD = 13
    SERVICE_LANE = 14
    SIDEWALK = 15
    BRIDGE = 16
    BUILDING = 17
    TUNNEL = 18
    PERSON = 19
    BICYCLIST = 20
    MOTORCYCLIST = 21
    OTHER_RIDER = 22
    LANE_MARKING_CROSSWALK = 23
    LANE_MARKING_GENERAL = 24
    MOUNTAIN = 25
    SAND = 26
    SKY = 27
    SNOW = 28
    TERRAIN = 29
    VEGETATION = 30
    WATER = 31
    BANNER = 32
    BENCH = 33
    BIKE_RACK = 34
    BILLBOARD = 35
    CATCH_BASIN = 36
    CCTV_CAMERA = 37
    FIRE_HYDRANT = 38
    JUNCTION_BOX = 39
    MAILBOX = 40
    MANHOLE = 41
    PHONE_BOOTH = 42
    POTHOLE = 43
    STREET_LIGHT = 44
    POLE = 45
    TRAFFIC_SIGN_FRAME = 46
    UTILITY_POLE = 47
    TRAFFIC_LIGHT = 48
    TRAFFIC_SIGN_BACK = 49
    TRAFFIC_SIGN_FRONT = 50
    TRASH_CAN = 51
    BICYCLE = 52
    BOAT = 53
    BUS = 54
    CAR = 55
    CARAVAN = 56
    MOTORCYCLE = 57
    ON_RAILS = 58
    OTHER_VEHICLE = 59
    TRAILER = 60
    TRUCK = 61
    WHEELED_SLOW = 62
    CAR_MOUNT = 63
    EGO_VEHICLE = 64

@functools.lru_cache(maxsize=1)
def get_cached_seg_former() -> "SegFormerMapillary":
    return SegFormerMapillary()


@functools.lru_cache(maxsize=40)
def cached_predict(img_path) -> np.ndarray:
    model = get_cached_seg_former()
    return model.predict(img_path)


def predict_with_array(img_arr) -> np.ndarray:
    model = get_cached_seg_former()
    return model.predict(img_arr)


class SegFormerMapillary:
    """
    example usage:
        model = SegFormerMapillary()
        pred = model.predict(<image-path>)
        road = model.get_mask(pred, CityScapesCls.ROAD)

        # simple numpy mask, you can overlay now
        orig = cv2.imread(<image-path>)
        # mark road as red, bgr
        orig[road] = [0, 0, 255]
        cv2.imshow(orig)
    """

    def __init__(self):
        # from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-semantic", use_fast=True
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-mapillary-vistas-semantic"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor
        self.model = model.to(self.device)

    def _predict(self, pil_image):
        img = pil_image
        orig = np.array(img)
        h, w, _ = orig.shape

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # you can pass them to processor for postprocessing
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]
        return predicted_semantic_map.cpu().numpy()


    def predict_bgr_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._predict(Image.fromarray(img))

    def predict(self, img_path):
        if isinstance(img_path, np.ndarray):
            img = Image.fromarray(img_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")
        # img = Image.open(img_path).convert("RGB")
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


def show_seg_mask(pred):
    label_by_name = {mem.value: mem.name for mem in Label}
    show_segmentation_pred(pred, label_by_name)