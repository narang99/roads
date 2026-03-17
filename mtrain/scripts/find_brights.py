from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from mtrain.smallnet.unet.extract.cropping import bbox_based
from mtrain.utils import show, overlay_mask_on_img as OV, DiskBooleanMask, DiskImage, mkdir
from pycocotools.coco import COCO
import json
from mtrain.tqdm import Progress
from tqdm import tqdm
from multiprocessing import Pool
import itertools
from mtrain.neg_mask.taco.extract import anns_to_mask
from collections import defaultdict
from mtrain.disk import DiskBooleanMask, DiskImage

DS = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/")
TACO = Path("/Users/hariomnarang/Desktop/personal/TACO/data")
ANN_FILE = TACO / "annotations.json"
EXT = DS / "T007-uncentered/data/EXT"
WHITE_DS = DS / "T007-uncentered" / "whitish"

coco = COCO(ANN_FILE)

def get_img_and_mask(image_path):
    image_path = Path(image_path)
    mask_path = EXT / "masks" / f"{image_path.stem}.png"
    return DiskImage.load(image_path), DiskBooleanMask.load(mask_path)


def get_cats(image_path, meta):
    image_path = Path(image_path)
    img_id = image_path.stem
    cats, sups = meta["img_id_to_cats"][img_id], meta["img_id_to_sups"][img_id]
    return cats, sups


def get_sup_to_img_id(meta):
    sup_to_img_id = defaultdict(list)
    for img_id, sups in meta["img_id_to_sups"].items():
        for sup in sups:
            sup_to_img_id[sup].append(img_id)
    return sup_to_img_id

def extract_crop_of_bbox(
    image_path,
    bbox: bbox_based.Bbox,
    target_bbox_height: int,
    target_crop_size: int,
):
    img, mask = get_img_and_mask(image_path)
    en_bbox, scaled_leftover_height, scaled_leftover_width = bbox_based.get_engulfing_bbox_to_resize(
        mask, bbox, target_bbox_height, target_crop_size, target_crop_size
    )
    resized_img = bbox_based.resize_bbox_in_img(
        img, en_bbox, target_crop_size, target_crop_size, scaled_leftover_height, scaled_leftover_width
    )
    resized_mask = bbox_based.resize_bbox_in_img(
        mask, en_bbox, target_crop_size, target_crop_size, scaled_leftover_height, scaled_leftover_width
    )

    return en_bbox, resized_img, resized_mask

def get_bbox_of_annotation(ann_id) -> bbox_based.Bbox:
    ann = coco.loadAnns(ann_id)[0]
    x, y, w, h = ann["bbox"]
    return bbox_based.Bbox(int(x),int(y),int(w),int(h))

def get_whiteness_ratio(rgb_image, mask):
    # Convert RGB to HSV (Note: cv2.COLOR_RGB2HSV)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    
    # Isolate pixels inside the segmentation mask
    object_pixels = hsv[mask > 0]
    
    if object_pixels.size == 0:
        return False
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 50, 255])
    white_mask = np.all((object_pixels >= lower_white) & (object_pixels <= upper_white), axis=1)
    white_pixel_count = np.sum(white_mask)
    whiteness_ratio = white_pixel_count / object_pixels.shape[0]

    return whiteness_ratio, white_pixel_count
    

def get_whiteness_ratio_hls(rgb_image, mask):
    # Convert RGB to HLS (Note: cv2.COLOR_RGB2HLS)
    # The order of channels in the result is [Hue, Lightness, Saturation]
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    
    # Isolate pixels inside the segmentation mask
    object_pixels = hls[mask > 0]
    
    if object_pixels.size == 0:
        return 0.0, 0
    
    # Define HLS thresholds for White:
    # 1. Hue: Range doesn't matter for pure white (0 to 179)
    # 2. Lightness: Should be very high (e.g., 200 to 255)
    # 3. Saturation: Should be very low (e.g., 0 to 50)
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([179, 255, 50])

    # Check which pixels fall within the range
    # np.all(..., axis=1) checks that H, L, and S all meet the criteria for each pixel
    white_mask = np.all((object_pixels >= lower_white) & (object_pixels <= upper_white), axis=1)
    
    # Count the True values
    white_pixel_count = np.sum(white_mask)
    
    # Total number of pixels in the mask
    total_pixels = object_pixels.shape[0]
    
    whiteness_ratio = white_pixel_count / total_pixels
    
    return whiteness_ratio, white_pixel_count

def is_whitish_hsv_rgb_v2(rgb_image, mask, s_thresh=40, v_thresh=200):
    # Convert RGB to HSV (Note: cv2.COLOR_RGB2HSV)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    
    # Isolate pixels inside the segmentation mask
    object_pixels = hsv[mask > 0]
    
    if object_pixels.size == 0:
        return False

    # Calculate averages
    # S (Index 1): 0 is white/gray, 255 is vivid color
    # V (Index 2): 0 is black, 255 is bright
    mean_s = np.mean(object_pixels[:, 1])
    mean_v = np.mean(object_pixels[:, 2])
    # Return True if it lacks color and is bright
    return mean_s < s_thresh and mean_v > v_thresh

def is_whitish_hsv_rgb(rgb_image, mask, s_thresh=40, v_thresh=200):
    # Convert RGB to HSV (Note: cv2.COLOR_RGB2HSV)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Isolate pixels inside the segmentation mask
    object_pixels = hsv[mask > 0]
    
    if object_pixels.size == 0:
        return False

    # Calculate averages
    # S (Index 1): 0 is white/gray, 255 is vivid color
    # V (Index 2): 0 is black, 255 is bright
    mean_s = np.mean(object_pixels[:, 1])
    mean_v = np.mean(object_pixels[:, 2])
    
    # Return True if it lacks color and is bright
    return mean_s < s_thresh and mean_v > v_thresh

def is_whitish_grayscale(image, mask, intensity_thresh=220, coverage_pct=0.7):
    # Convert BGR to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Extract object pixels
    object_pixels = gray[mask > 0]
    
    if object_pixels.size == 0:
        return False
    
    # Count how many pixels are above our brightness threshold
    white_pixel_count = np.sum(object_pixels > intensity_thresh)
    white_fraction = white_pixel_count / object_pixels.size
    
    # Logic: If 70% of the object is very bright, call it white
    return white_fraction > coverage_pct

def is_bright_object(rgb_image, mask, v_threshold=200):
    # Convert to HSV to isolate the 'Value' (Brightness) channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Get pixels of the object
    object_v_channel = hsv[mask > 0][:, 2]
    
    if object_v_channel.size == 0:
        return False
        
    # Check if the average brightness exceeds the threshold
    return np.mean(object_v_channel) > v_threshold


def get_whiteish_images(image_paths):
    for image_path in image_paths:
        img, _ = get_img_and_mask(image_path)
        img_id = int(image_path.stem)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        should_add_ann_ids = []
        for ann_id, ann in zip(ann_ids, anns):
            ann_id = int(ann_id)
            mask = anns_to_mask(coco.loadAnns(ann_id), img.shape[0], img.shape[1])
            ratio, _ = get_whiteness_ratio(img, mask)
            if ratio > 0.2 and (is_whitish_grayscale(img, mask) or is_whitish_hsv_rgb(img, mask) or is_bright_object(img, mask, 180)):
                should_add_ann_ids.append(ann_id)
        if should_add_ann_ids:
            yield image_path, should_add_ann_ids


class WhiteImageSaver:
    def __init__(self, out_images, out_masks):
        self.out_images, self.out_masks = out_images, out_masks

    def __call__(self, image_paths):
        progress = Progress(len(image_paths))
        for i, (image_path, ann_ids) in enumerate(get_whiteish_images(image_paths)):
            image = DiskImage.load(image_path)
            # value=1 -> forces 1 for all cats, binary segmentation only
            mask = anns_to_mask(coco.loadAnns(ann_ids), image.shape[0], image.shape[1], value=1)

            fname = image_path.stem
            DiskImage.save(image, self.out_images / f"{fname}.jpg")
            DiskBooleanMask.save(mask, self.out_masks / f"{fname}.png")
            progress(i)

def save_whiteish_images(coco, images_dir, out_dir):
    out_dir = mkdir(out_dir)
    out_images = mkdir(out_dir / "images")
    out_masks = mkdir(out_dir / "masks")

    def _do_for_image_paths(image_paths):
        progress = Progress(len(image_paths))
        for i, (image_path, ann_ids) in enumerate(get_whiteish_images(image_paths)):
            image = DiskImage.load(image_path)
            # value=1 -> forces 1 for all cats, binary segmentation only
            mask = anns_to_mask(coco.loadAnns(ann_ids), image.shape[0], image.shape[1], value=1)

            fname = image_path.stem
            DiskImage.save(image, out_images / f"{fname}.jpg")
            DiskBooleanMask.save(mask, out_masks / f"{fname}.png")
            progress(i)


    numworkers = 4
    with Pool(numworkers) as p:
        runner = WhiteImageSaver(out_images, out_masks)
        image_paths = list(images_dir.glob("*.jpeg"))
        chunk_size = len(image_paths) // numworkers
        print("chunk size", chunk_size)
        p.map(runner, itertools.batched(image_paths, chunk_size))


if __name__ == '__main__':
    save_whiteish_images(coco, EXT / "images", WHITE_DS / "ext")