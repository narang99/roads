from pathlib import Path
import functools
import numpy as np
import cv2
from fastai.vision.all import (
    load_learner,
    vision_learner,
    resnet18,
    accuracy,
    F1Score,
    CrossEntropyLossFlat,
    ProgressCallback,
)
from mtrain.disk import DiskImage, DiskBooleanMask
from mtrain.seg import mapillary as mapi, elevated_vegetation as elev
from mtrain.neg_mask.model.learner import dummy_dls
from mtrain.neg_mask.model.predict import trash as pred_trash, flower as pred_flower
from mtrain.smallnet.unet.predict.strided import single

SMALLNET_SIZE = 100
SMALLNET_STRIDES = [50]
SMALLNET_AREA_THRES = 5

NEGMASK_SIZE = 130
FLOWER_CROP_SIZE = 60

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


@functools.lru_cache(maxsize=1)
def get_default_smallnet_learner():
    """Load default smallnet learner from gen_preds.py path"""
    SMALLNET_MODEL_PATH = "/Users/hariomnarang/Desktop/gdrive-sync/garbage/experiments/enguled-bbox-levels-crops-v3/log/export_iter_14.pkl"
    return load_learner(SMALLNET_MODEL_PATH)


@functools.lru_cache(maxsize=1)
def get_default_negmask_learner():
    """Load default negmask learner from gen_preds.py path and setup"""
    print("yaahhhahahah")
    NEG_MASK_MODEL_PATH = Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/models/trash_classification/resnet18-size_130-chan_8-with_augs-iter_30"
    )
    LABELS = ["other", "trash"]
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


@functools.lru_cache(maxsize=1)
def get_default_flower_learner():
    """Load default flower learner"""
    FLOWER_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/flowers/inat/train/size-100_crop-60/models/with-aug-tfms-iter-15.pkl"
    return load_learner(FLOWER_MODEL_PATH)


class ExampleDir:
    def __init__(
        self,
        direc: Path | str,
        smallnet_learner=None,
        negmask_learner=None,
        flower_learner=None,
        mapillary_segformer=None,
        elev_segformer=None,
    ):
        self.d = Path(direc)
        if not (self.d / "image.jpg").exists():
            raise Exception(f"image {self.d / 'image.jpg'} does not exist")

        self.smallnet_learner = (
            smallnet_learner
            if smallnet_learner is not None
            else get_default_smallnet_learner()
        )
        self.negmask_learner = (
            negmask_learner
            if negmask_learner is not None
            else get_default_negmask_learner()
        )
        self.flower_learner = (
            flower_learner
            if flower_learner is not None
            else get_default_flower_learner()
        )
        self.mapillary_segformer = mapillary_segformer
        self.elev_segformer = elev_segformer

    @property
    def image_path(self):
        return self.d / "image.jpg"

    def _load_and_resize_image(self):
        """Load and resize image if needed (from gen_preds.py logic)"""
        img_arr = DiskImage.load(self.image_path)
        if img_arr.shape[0] > 1024 or img_arr.shape[1] > 1024:
            img_arr = cv2.resize(img_arr, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        return img_arr

    def _generate_smallnet_mask(self):
        """Generate initial mask using smallnet model"""

        img_arr = self._load_and_resize_image()
        mask = single.strided_predict_unet_only_mask(
            img_arr, SMALLNET_SIZE, self.smallnet_learner, SMALLNET_STRIDES
        )
        DiskBooleanMask.save(mask, self.d / "mask.png")
        return mask

    def _generate_mapi_mask(self):
        """Generate mapillary segmentation mask"""
        mapi_pred = mapi.cached_predict(self.image_path)
        DiskBooleanMask.save(mapi_pred.astype(np.uint8), self.d / "mapi.png")
        return mapi_pred

    def _generate_elev_mask(self):
        """Generate elevated vegetation segmentation mask"""
        if self.elev_segformer is not None:
            elev_pred = elev.cached_predict(self.image_path, self.elev_segformer)
        else:
            elev_pred = elev.cached_predict(self.image_path)
        DiskBooleanMask.save(elev_pred.astype(np.uint8), self.d / "elev.png")
        return elev_pred

    def _extract_regions(self, mask: np.ndarray) -> list[dict]:
        """Extract connected components from binary mask (from gen_preds.py)"""
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

    def _get_mask_with_area_greater(self, mask, threshold):
        """Filter mask by area threshold (from gen_preds.py)"""
        mask = mask.astype(bool)
        regions = self._extract_regions(mask)
        regions = [r for r in regions if r["area"] >= threshold]
        res = np.zeros(mask.shape, bool)
        for r in regions:
            res |= r["component_mask"].astype(bool)
        return res

    def _get_trimmed_mask(self, mask, elev_pred, mapi_pred):
        """Trim mask by removing unwanted segments (from gen_preds.py)"""
        if mapi_pred is not None:
            mapi_exclude_mask = mapi.get_mask_with_labels(
                mapi_pred, MAPI_LABELS_TO_EXCLUDE
            )
        else:
            mapi_exclude_mask = np.zeros(mask.shape, dtype=bool)

        if elev_pred is not None:
            elev_exclude_mask = elev.get_mask_with_labels(
                elev_pred, ELEV_LABELS_TO_EXCLUDE
            )
        else:
            elev_exclude_mask = np.zeros(mask.shape, dtype=bool)

        return mask & (~mapi_exclude_mask) & (~elev_exclude_mask)

    def _generate_trimmed_mask(self):
        """Generate final trimmed mask (m2.png)"""
        mask = DiskBooleanMask.load(self.smallnet_mask_path())
        elev_pred = DiskBooleanMask.load(self.elev_mask_path())
        mapi_pred = DiskBooleanMask.load(self.mapi_mask_path())

        trimmed = self._get_trimmed_mask(mask, elev_pred, mapi_pred)
        trimmed = self._get_mask_with_area_greater(trimmed, SMALLNET_AREA_THRES)

        DiskBooleanMask.save(trimmed, self.d / "m2.png")
        return trimmed

    def _generate_negmask_probs(self):
        """Generate trash/other probability arrays"""

        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_mask_path())

        other_probs, trash_probs = pred_trash.predict_and_return_prob_masks(
            image, mask, self.negmask_learner, NEGMASK_SIZE
        )

        np.save(self.d / "trash_probs.npy", trash_probs)
        np.save(self.d / "other_probs.npy", other_probs)

        return trash_probs, other_probs

    def _generate_flower_probs(self):
        """Generate flower pos/neg probability arrays using 3-channel ResNet"""

        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_mask_path())

        flower_neg_probs, flower_pos_probs = pred_flower.predict_and_return_prob_masks(
            image, mask, self.flower_learner, FLOWER_CROP_SIZE
        )

        np.save(self.d / "flower_pos_probs.npy", flower_pos_probs)
        np.save(self.d / "flower_neg_probs.npy", flower_neg_probs)

        return flower_pos_probs, flower_neg_probs

    # Public functions that return paths and generate if missing
    def smallnet_mask_path(self, force=False):
        """Return path to smallnet mask, generate if missing"""
        path = self.d / "mask.png"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self._generate_smallnet_mask()
        return path

    def mapi_mask_path(self, force=False):
        """Return path to mapillary mask, generate if missing"""
        path = self.d / "mapi.png"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self._generate_mapi_mask()
        return path

    def elev_mask_path(self, force=False):
        """Return path to elevation mask, generate if missing"""
        path = self.d / "elev.png"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self._generate_elev_mask()
        return path

    def trimmed_mask_path(self, force=False):
        """Return path to trimmed mask (m2.png), generate if missing"""
        path = self.d / "m2.png"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.smallnet_mask_path()
            self.mapi_mask_path()
            self.elev_mask_path()
            self._generate_trimmed_mask()
        return path

    def trash_probs_path(self, force=False):
        """Return path to trash probabilities, generate if missing"""
        path = self.d / "trash_probs.npy"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self._generate_negmask_probs()
        return path

    def other_probs_path(self, force=False):
        """Return path to other probabilities, generate if missing"""
        path = self.d / "other_probs.npy"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.smallnet_mask_path()
            self._generate_negmask_probs()
        return path

    def flower_pos_probs_path(self, force=False):
        """Return path to flower positive probabilities, generate if missing"""
        path = self.d / "flower_pos_probs.npy"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.trimmed_mask_path()
            self._generate_flower_probs()
        return path

    def flower_neg_probs_path(self, force=False):
        """Return path to flower negative probabilities, generate if missing"""
        path = self.d / "flower_neg_probs.npy"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.trimmed_mask_path()
            self._generate_flower_probs()
        return path

    # def final_mask(self, trash_thres=0, flower_thres=0):
    #     # always precomputes the last part, always "forced" basically
    #     mask = self.trimmed_mask_path()
    #     trash = self.trash_probs_path()
    #     other = self.other_probs_path()

    #     flower_pos = self.flower_pos_probs_path()
    #     flower_neg = self.flower_neg_probs_path()

    #     flower_pos, flower_neg = np.load(flower_pos), np.load(flower_neg)
    #     above_neg, _, _ = get_flower_mask_above_threshold_and_neg(
    #         flower_pos, flower_neg, flower_thres
    #     )

    #     trash, other = np.load(trash), np.load(other)
    #     above_other, _, _ = get_trash_mask_above_threshold_and_other(
    #         trash, other, trash_thres
    #     )

    #     mask = DiskBooleanMask.load(mask)

    #     return mask.astype(bool) & above_neg.astype(bool) & above_other.astype(bool)
