from PIL import Image, ExifTags
import functools
import numpy as np
import cv2


def extract_mask_for_image_id(img_id, coco, taco_dir):
    image_path = taco_dir / coco.loadImgs(img_id)[0]["file_name"]
    annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
    anns_sel = coco.loadAnns(annIds)
    img_array = load_image(image_path)
    h, w = img_array.shape[:2]
    mask = anns_to_mask(anns_sel, h, w)
    return img_array, mask


def load_image(image_path):
    # Obtain Exif orientation tag code
    orientation = get_orientation_tag()

    img = Image.open(image_path)

    # Load and process image metadata
    if img._getexif() and orientation:
        exif = dict(img._getexif().items())
        # Rotate portrait and upside down images if necessary
        if orientation in exif:
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            if exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            if exif[orientation] == 8:
                img = img.rotate(90, expand=True)

    img = img.convert("RGB")
    return np.array(img)


@functools.lru_cache(maxsize=1)
def get_orientation_tag():
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            return orientation
    return None


def anns_to_mask(anns_sel, height, width, value=1):
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns_sel:
        for seg in ann["segmentation"]:
            poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], value)
    return mask
