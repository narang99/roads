from mtrain.neg_mask.openai_clip import get_images_from_clip_file
from itertools import batched
from mtrain.neg_mask.model.predict.full_image_8chan import (
    predict_and_return_probs,
)
from fastai.vision.all import (
    load_learner,
    vision_learner,
    DataLoaders,
    resnet18,
    accuracy,
    F1Score,
    CrossEntropyLossFlat,
    ProgressCallback,
)
from tqdm import tqdm
import numpy as np
import cv2
from mtrain.disk import DiskImage, DiskBooleanMask
from mtrain.seg import mapillary as mapi, elevated_vegetation as elev
from mtrain.neg_mask.model.learner import dummy_dls
from mtrain.neg_mask.model.crop_level_dataset import CropLevelDataset2Chan
import shutil
from mtrain.smallnet.unet.predict.strided import single, multiple
from pathlib import Path

################################# input images parameters #################################################
############# users should change this ############################
# IMAGES = list(
#     Path("/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/personal").rglob("image.jpg")
# )
CLIP_NAME = "delhi_litter"
CLIP_FILE = f"/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/trash/clip_{CLIP_NAME}.txt"
CLS_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/crop_level")


TOTAL_SAMPLES = 100
DEST_DIR = Path(
    "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/full_image_runs/delhi_litter"
)

def get_images_not_in_train_set():
    images = get_images_from_clip_file(CLIP_FILE)
    all_dirs = set()
    for label in ["other", "trash"]:
        sample_names_without_crop_idx = (p.name.split("_")[0] for p in (CLS_DIR / label).glob("*") if p.is_dir())
        for s in sample_names_without_crop_idx:
            all_dirs.add(s)
    return [img for img in images if img.stem not in all_dirs]
    

IMAGES = list(get_images_not_in_train_set())[:TOTAL_SAMPLES]
##########################################################################################################


############### model definitions ############################
SMALLNET_MODEL_PATH = "/Users/hariomnarang/Desktop/gdrive-sync/garbage/experiments/enguled-bbox-levels-crops-v3/log/export_iter_14.pkl"
SMALLNET_SIZE = 100
SMALLNET_STRIDES = [50]
SMALLNET_AREA_THRES = 5

NEG_MASK_MODEL_PATH = Path(
    "/Users/hariomnarang/Desktop/personal/roads/datasets/models/trash_classification/resnet18-size_220-chan_8-with_augs-iter_15"
)
################## done #######################################


##################### segmentation parameters ###########################
MAPI_LABELS_TO_EXCLUDE = [
    mapi.Label.PERSON,
    mapi.Label.MOTORCYCLIST,
    mapi.Label.BICYCLIST,
    mapi.Label.GROUND_ANIMAL,
    mapi.Label.OTHER_RIDER,
    mapi.Label.BIRD,
    mapi.Label.SKY,
    mapi.Label.BOAT,
    mapi.Label.BUS,
    mapi.Label.CAR,
    mapi.Label.CARAVAN,
    mapi.Label.MOTORCYCLE,
    mapi.Label.ON_RAILS,
    mapi.Label.OTHER_VEHICLE,
    mapi.Label.EGO_VEHICLE,
    mapi.Label.TRAILER,
    mapi.Label.TRUCK,
    mapi.Label.WHEELED_SLOW,
    mapi.Label.CAR_MOUNT,
    mapi.Label.BICYCLE,
    mapi.Label.BRIDGE,
    mapi.Label.TUNNEL,
    mapi.Label.BUILDING,
    mapi.Label.BILLBOARD,
    mapi.Label.BANNER,
    mapi.Label.STREET_LIGHT,
    mapi.Label.JUNCTION_BOX,
    mapi.Label.MAILBOX,
    mapi.Label.MOUNTAIN,
    mapi.Label.PHONE_BOOTH,
    mapi.Label.TRAFFIC_SIGN_FRONT,
    mapi.Label.TRAFFIC_SIGN_FRAME,
    mapi.Label.TRAFFIC_SIGN_BACK,
]

ELEV_LABELS_TO_EXCLUDE = [elev.Label.ELEVATED_VEGETATION]
##################### segmentation parameters ###########################


def save_filter_masks(dirs: list[Path]):
    print("STAGE: filter masks")
    for d in tqdm(dirs):
        img = d / "image.jpg"
        if not (d / "mapi.png").exists():
            mapi_pred = mapi.cached_predict(img)
            DiskBooleanMask.save(mapi_pred.astype(np.uint8), d / "mapi.png")
        if not (d / "elev.png").exists():
            elev_pred = elev.cached_predict(img)
            DiskBooleanMask.save(elev_pred.astype(np.uint8), d / "elev.png")


def _load_and_resize_single_image(image_path):
    img_arr = DiskImage.load(image_path)
    if img_arr.shape[0] > 1024 or img_arr.shape[1] > 1024:
        img_arr = cv2.resize(img_arr, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    return img_arr


def predict_pass1(images, model_path, size, strides, dest_dir):
    learner = load_learner(model_path)
    dirs_created = []

    print("STAGE: First pass preds")
    for i, img in tqdm(enumerate(list(images))):
        img = Path(img)

        img_arr = _load_and_resize_single_image(img)

        mask = single.strided_predict_unet_only_mask(img_arr, size, learner, strides)
        dest = dest_dir / img.stem
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        DiskImage.save(img_arr, dest / "image.jpg")
        DiskBooleanMask.save(mask, dest / "mask.png")

        dirs_created.append(dest)
    return dirs_created


def extract_regions(mask: np.ndarray) -> list[dict]:
    """
    Extract connected components from a binary mask.
    Returns list of dicts with: label, area, bbox, centroid, contour
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    regions = []
    for i in range(1, num_labels):  # skip background (0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions.append(
            {
                "label": i,
                "area": int(area),
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": (float(cx), float(cy)),
                "contour": contours[0] if contours else None,
                "component_mask": component_mask,
            }
        )
    return regions


def get_mask_with_area_greater(mask, threshold):
    mask = mask.astype(bool)
    regions = extract_regions(mask)
    regions = [r for r in regions if r["area"] >= threshold]
    res = np.zeros(mask.shape, bool)
    for r in regions:
        res |= r["component_mask"].astype(bool)
    return res


def save_trimmed_masks(dirs):
    print("STAGE: Mask trimming")
    for d in dirs:
        try:
            mask = DiskBooleanMask.load(d / "mask.png")
            elev_pred = DiskBooleanMask.load(d / "elev.png")
            mapi_pred = DiskBooleanMask.load(d / "mapi.png")
            mask = get_trimmed_mask(mask, elev_pred, mapi_pred)
            mask = get_mask_with_area_greater(mask, SMALLNET_AREA_THRES)
            DiskBooleanMask.save(mask, d / "m2.png")
        except Exception as ex:
            print(f"WARN: failure in saving trimmed mask for d={d} cause={ex}")


def get_trimmed_mask(mask, elev_pred, mapi_pred):
    # for now, we remove all obvious things that we see
    # then we will go through all the remaining masks
    # in decreasing order of the amount of segmentation
    # and remove more stuff
    if mapi_pred is not None:
        mapi_exclude_mask = mapi.get_mask_with_labels(mapi_pred, MAPI_LABELS_TO_EXCLUDE)
    else:
        mapi_exclude_mask = np.zeros(mask.shape, dtype=bool)
    if elev_pred is not None:
        elev_exclude_mask = elev.get_mask_with_labels(elev_pred, ELEV_LABELS_TO_EXCLUDE)
    else:
        elev_exclude_mask = np.zeros(mask.shape, dtype=bool)

    return mask & (~mapi_exclude_mask) & (~elev_exclude_mask)


def _load_neg_mask_model(model_path):
    LABELS = ["other", "trash"]
    DataLoaders.from_dsets(
        CropLevelDataset2Chan([], LABELS, True, medium_pad=220),
        CropLevelDataset2Chan([], LABELS, False, medium_pad=220),
    )

    learner = vision_learner(
        dummy_dls(LABELS),
        resnet18,
        n_in=8,
        metrics=[accuracy, F1Score(average="macro")],
        loss_func=CrossEntropyLossFlat(),
        n_out=len(LABELS),
        normalize=False,
    )
    learner = learner.remove_cb(ProgressCallback)

    learner = learner.load(NEG_MASK_MODEL_PATH)
    return learner


def save_neg_mask_probs(pred_dirs):
    print(f"STAGE: negmask, total directories={len(pred_dirs)}")
    learner = _load_neg_mask_model(NEG_MASK_MODEL_PATH)
    for d in tqdm(pred_dirs):
        image, mask = (
            DiskImage.load(d / "image.jpg"),
            DiskBooleanMask.load(d / "mask.png"),
        )
        trash_probs, other_probs = predict_and_return_probs(
            image,
            mask,
            learner,
            220,
        )
        np.save(d / "trash_probs.npy", trash_probs)
        np.save(d / "other_probs.npy", other_probs)


def main():
    pred_dirs = predict_pass1(
        IMAGES, SMALLNET_MODEL_PATH, SMALLNET_SIZE, SMALLNET_STRIDES, DEST_DIR
    )
    save_filter_masks(pred_dirs)
    save_trimmed_masks(pred_dirs)
    save_neg_mask_probs(pred_dirs)


if __name__ == "__main__":
    main()
