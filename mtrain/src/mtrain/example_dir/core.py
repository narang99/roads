from sklearn.externals.array_api_compat.torch import neg
import torch
from functools import partial
from fastai.basics import (
    DataLoaders,
    default_device,
    Precision,
)
from mtrain.neg_mask.model.datasets.blur_pad_dl import BlurPadDataset, CropTfmsOutsideBbox, blur_overwriter
from pathlib import Path
import functools
import numpy as np
import cv2
from torch import functional as F
from fastai.vision.all import (
    load_learner,
    vision_learner,
    resnet18,
    get_image_files,
    CrossEntropyLossFlat,
    ProgressCallback,
    SegmentationDataLoaders,
    Resize,
    xresnet18,
    unet_learner,
)
from mtrain.disk import DiskImage, DiskBooleanMask
from mtrain.seg import mapillary as mapi, elevated_vegetation as elev
from mtrain.neg_mask.model.predict import trash as pred_trash, flower as pred_flower
from mtrain.smallnet.unet.predict.strided import single

SMALLNET_SIZE = 100
SMALLNET_STRIDES = [50]
SMALLNET_AREA_THRES = 5

NEGMASK_SIZE = 224
NEGMASK_BLUR_KERNEL_SZ = 13
NEGMASK_BLUR_KERNEL_SIGMA = 4
NEGMASK_BBOX_PAD = 10
NEGMASK_GAUSSIAN_STEP_DOWN_MIN_VALUE = 0.3

NEGMASK_UNBLURRED_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/unblurred/v2/blurpad/fullset-p10-iter100-verified.pt"
# NEGMASK_STEP_DOWN_GAUSSIAN_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/successive-unblur/tfm-gaussstepdown_min-3_samples-all-xresnet18_iter-50.pth"
NEG_MASKSTEP_EDGE_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/successive-unblur/tfm-stepdown_ratio-5_samples-all-xresnet18_iter-20.pth"
# NEGMASK_BASELINE = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/neg-baseline/baseline-pad-10-crop-size-64-iter-20.pt"
# NEGMASK_224_STEP_EDGE = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/successive-224/st_ed_tfm0-final-all-data-iter-20.pth"
NEGMASK_224_STEP_EDGE = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/successive-224/st_ed_tfm0-final-all-data-with-taco-iter-10.pth"

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


def load_npz(path):
    with np.load(path) as compressed:
        return compressed['data']

def save_npz(path, data):
    np.savez_compressed(path, data=data)

@functools.lru_cache(maxsize=1)
def get_default_smallnet_learner():
    """Load default smallnet learner from gen_preds.py path"""
    SMALLNET_MODEL_PATH = "/Users/hariomnarang/Desktop/gdrive-sync/garbage/experiments/enguled-bbox-levels-crops-v3/log/export_iter_14.pkl"
    return load_learner(SMALLNET_MODEL_PATH)


@functools.lru_cache(maxsize=1)
def get_default_smallnet_50x50_learner():
    SMALLNET_50x50_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/smallnet_50x50/iter80.pkl"
    return load_learner(SMALLNET_50x50_MODEL_PATH)


@functools.lru_cache(maxsize=1)
def get_default_smallnet_50x50_learner_using_raw_torch_pth():
    """Load default smallnet learner from gen_preds.py path

    fastai sucks at this. environment mismatches, or version mismatches etc break the model from colab <-> local
    It is so unpredictable that i need to do a show_results once in every new environment since it keeps breaking
    the best way is to use torch directly, but I don't know how to right now. You always need to instantiate the learner too
    instantiating takes a dataloader, which needs ACTUAL EXISTING data (even for inference mode)
    ive exported pkl model in my local manually, for now use that, keeping this in case needed for future
    """
    SMALLNET_50x50_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/smallnet_50x50/raw_torch_iter80.pth"

    DATA_DIR = Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/T007-uncentered/whitish/small_crops"
    )

    dummy_dls = SegmentationDataLoaders.from_label_func(
        "./",
        bs=4,
        fnames=get_image_files(DATA_DIR / "images"),
        label_func=lambda o: DATA_DIR / "masks" / f"{o.stem}.png",
        codes=np.array(["background", "trash"]),
        item_tfms=Resize(64),
        num_workers=8,
        persistent_workers=True,
    )
    learner = unet_learner(dummy_dls, xresnet18, n_out=2, pretrained=True)

    learner = learner.remove_cb(ProgressCallback)
    state_dict = torch.load(SMALLNET_50x50_MODEL_PATH, map_location=default_device())
    learner.model.load_state_dict(state_dict, strict=True)
    learner.model.eval()
    return learner


def dummy_unblur_dls():
    train_ds = BlurPadDataset([], Path("./masks"), 128, False)
    valid_ds = BlurPadDataset([], Path("./masks"), 128, True)
    dls = DataLoaders.from_dsets(
        train_ds,
        valid_ds,
        device=default_device(),
        num_workers=4,
        bs=16,
        persistent_workers=True,
    )
    return dls


@functools.lru_cache(maxsize=5)
def get_default_negmask_learner(model_path, arch="xresnet18"):
    arch = xresnet18 if arch == "xresnet18" else resnet18
    # dummy dls
    dls = dummy_unblur_dls()
    learn = vision_learner(
        dls,
        arch,
        metrics=[Precision()],
        loss_func=CrossEntropyLossFlat(),
        n_out=2,
        normalize=False,
        n_in=3,
        pretrained=True,
    )

    # load statedict
    learn = learn.remove_cb(ProgressCallback)
    state_dict = torch.load(model_path, map_location=default_device())
    learn.model.load_state_dict(state_dict, strict=True)
    learn.model.eval()
    return learn


@functools.lru_cache(maxsize=1)
def get_default_flower_learner():
    """Load default flower learner"""
    FLOWER_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/flowers/inat/train/size-100_crop-60/models/with-aug-tfms-iter-15.pkl"
    return load_learner(FLOWER_MODEL_PATH)


def step_down_edge_tfm(cropped_image, mask, inner_bbox, ratio):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down(ratio)
    return tfm.crop

def _unlinked_if_force(p, force):
    p = Path(p)
    if p.exists() and force:
        p.unlink()
    return p

def loaded_mask_name_by_masks(negmask_mask_name_by_path):
    res = {}
    for mask_name in negmask_mask_name_by_path:
        trash = load_npz(negmask_mask_name_by_path[mask_name]["trash"])
        other = load_npz(negmask_mask_name_by_path[mask_name]["other"])
        res[mask_name] = {"trash": trash, "other": other}
    return res

class ExampleDir:
    """Abstraction of the single directory in inference for the full inference pipeline

    We have stages in inference, this class provides all stages for each directory
    The directory is called "example" here
    This class maintains the code for running inference and persisting it at correct places

    There is one problem though, it does not provide a lot of testing without deleting older results
    I will need to add a simpler API for simply getting the inference along with the original one to test
    """

    def __init__(
        self,
        direc: Path | str,
        smallnet_learner=None,
        smallnet_50x50_learner=None,
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
        self.smallnet_50x50_learner = (
            smallnet_50x50_learner
            if smallnet_50x50_learner is not None
            else get_default_smallnet_50x50_learner()
        )
        self.negmask_learners = {
            "step_edge_224": get_default_negmask_learner(NEGMASK_224_STEP_EDGE, "xresnet18"),
            "step_edge": get_default_negmask_learner(NEG_MASKSTEP_EDGE_MODEL_PATH, "xresnet18"),
            "unblurred": get_default_negmask_learner(NEGMASK_UNBLURRED_MODEL_PATH, "resnet18"),
        }
        # self.negmask_learner = (
        #     negmask_learner
        #     if negmask_learner is not None
        #     else get_default_negmask_learner(NEGMASK_224_STEP_EDGE)
        # )
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

    def load_and_resize_image(self):
        """Load and resize image if needed (from gen_preds.py logic)"""
        img_arr = DiskImage.load(self.image_path)
        if img_arr.shape[0] > 1024 or img_arr.shape[1] > 1024:
            img_arr = cv2.resize(img_arr, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        return img_arr

    def _generate_smallnet_mask(self):
        """Generate initial mask using smallnet model"""

        img_arr = self.load_and_resize_image()
        mask = single.strided_predict_unet_only_mask(
            img_arr, SMALLNET_SIZE, self.smallnet_learner, SMALLNET_STRIDES
        )
        DiskBooleanMask.save(mask, self.d / "mask.png")
        return mask

    def _generate_50x50_smallnet_mask(self):
        img_arr = self.load_and_resize_image()
        mask = single.strided_predict_unet_only_mask(
            img_arr, 64, self.smallnet_50x50_learner, [32]
        )
        DiskBooleanMask.save(mask, self.d / "mask_50x50.png")
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

    def _generate_50x50_trimmed_mask(self):
        """Generate final trimmed mask (m2_50x50.png)"""
        mask = DiskBooleanMask.load(self.smallnet_50x50_path())
        elev_pred = DiskBooleanMask.load(self.elev_mask_path())
        mapi_pred = DiskBooleanMask.load(self.mapi_mask_path())

        trimmed = self._get_trimmed_mask(mask, elev_pred, mapi_pred)
        trimmed = self._get_mask_with_area_greater(trimmed, SMALLNET_AREA_THRES)

        DiskBooleanMask.save(trimmed, self.d / "m2_50x50.png")
        return trimmed

    def _generate_trimmed_mask(self):
        """Generate final trimmed mask (m2.png)"""
        mask = DiskBooleanMask.load(self.smallnet_mask_path())
        elev_pred = DiskBooleanMask.load(self.elev_mask_path())
        mapi_pred = DiskBooleanMask.load(self.mapi_mask_path())

        trimmed = self._get_trimmed_mask(mask, elev_pred, mapi_pred)
        trimmed = self._get_mask_with_area_greater(trimmed, SMALLNET_AREA_THRES)

        DiskBooleanMask.save(trimmed, self.d / "m2.png")
        return trimmed

    def generate_50x50_negmask_probs(self):
        """Generate trash/other probability arrays"""

        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_50x50_mask_path())
        step_downer = partial(step_down_edge_tfm, ratio=0.5)

        other_probs, trash_probs = (
            pred_trash.predict_and_return_prob_masks_using_unblurred(
                image,
                mask,
                self.negmask_learners["step_edge_224"],
                crop_size=224,
                bbox_pad=10,
                mutator=step_downer,
            )
        )

        save_npz(self.d / "trash_probs_50x50.npz", trash_probs)
        save_npz(self.d / "other_probs_50x50.npz", other_probs)

        return trash_probs, other_probs

    def _gen_all_negmasks(self, image, mask):
        step_downer = partial(step_down_edge_tfm, ratio=0.5)
        step_other, step_trash = pred_trash.predict_and_return_prob_masks_using_unblurred(
            image,
            mask,
            self.negmask_learners["step_edge"],
            128,
            bbox_pad=10,
            mutator=step_downer,
        )
        unblur_other, unblur_trash = pred_trash.predict_and_return_prob_masks_using_unblurred(
            image,
            mask,
            self.negmask_learners["unblurred"],
            130,
            bbox_pad=10,
            mutator=blur_overwriter(13, 4),
        )
        step_224_other, step_224_trash = pred_trash.predict_and_return_prob_masks_using_unblurred(
            image,
            mask,
            self.negmask_learners["step_edge_224"],
            224,
            bbox_pad=10,
            mutator=step_downer,
        )
        return {
            "step_edge": (step_other, step_trash),
            "unblurred": (unblur_other, unblur_trash),
            "step_edge_224": (step_224_other, step_224_trash),
        }


    def generate_negmask_probs(self):
        """Generate trash/other probability arrays"""

        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_mask_path())

        step_downer = partial(step_down_edge_tfm, ratio=0.5)
        other_probs, trash_probs = (
            pred_trash.predict_and_return_prob_masks_using_unblurred(
                image,
                mask,
                self.negmask_learners["step_edge_224"],
                crop_size=224,
                bbox_pad=10,
                mutator=step_downer,
            )
        )

        save_npz(self.d / "trash_probs.npz", trash_probs)
        save_npz(self.d / "other_probs.npz", other_probs)

        return trash_probs, other_probs

    def _generate_flower_probs(self):
        """Generate flower pos/neg probability arrays using 3-channel ResNet"""

        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_mask_path())

        flower_neg_probs, flower_pos_probs = pred_flower.predict_and_return_prob_masks(
            image, mask, self.flower_learner, FLOWER_CROP_SIZE
        )

        save_npz(self.d / "flower_pos_probs.npz", flower_pos_probs)
        save_npz(self.d / "flower_neg_probs.npz", flower_neg_probs)

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

    def smallnet_50x50_path(self, force=False):
        path = self.d / "mask_50x50.png"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self._generate_50x50_smallnet_mask()
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

    def trimmed_50x50_mask_path(self, force=False):
        """Return path to trimmed mask (m2.png), generate if missing"""
        path = self.d / "m2_50x50.png"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.smallnet_50x50_path()
            self.mapi_mask_path()
            self.elev_mask_path()
            self._generate_50x50_trimmed_mask()
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

    def trash_50x50_probs_path(self, force=False):
        """Return path to trash probabilities, generate if missing"""
        path = self.d / "trash_probs_50x50.npz"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self.generate_50x50_negmask_probs()
        return path

    def other_50x50_probs_path(self, force=False):
        """Return path to other probabilities, generate if missing"""
        path = self.d / "other_probs_50x50.npz"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self.generate_50x50_negmask_probs()
        return path

    def _get_all_100x100_negmask_paths(self, mask_names):
        res = {}
        for mask_name in mask_names:
            res[mask_name] = {
                "other": self. d / f"other_probs_100x100_{mask_name}.npz",
                "trash": self. d / f"trash_probs_100x100_{mask_name}.npz",
            }
        return res

    def _get_all_50x50_negmask_paths(self, mask_names):
        res = {}
        for mask_name in mask_names:
            res[mask_name] = {
                "other": self. d / f"other_probs_50x50_{mask_name}.npz",
                "trash": self. d / f"trash_probs_50x50_{mask_name}.npz",
            }
        return res

    def _all_negmasks_exist(self, mask_name_by_paths, force):
        paths = []
        for mask_name, its_paths in mask_name_by_paths.items():
            trash_path = its_paths["trash"]
            other_path = its_paths["other"]
            paths.append(_unlinked_if_force(trash_path, force))
            paths.append(_unlinked_if_force(other_path, force))
        all_paths_exist = all([p.exists() for p in paths])
        return all_paths_exist

    def _persist_negmasks(self, all_masks, mask_name_by_paths):
        for mask_name, (mask_other, mask_trash) in all_masks.items():
            trash_path = mask_name_by_paths[mask_name]["trash"]
            other_path = mask_name_by_paths[mask_name]["other"]

            if not trash_path.exists():
                save_npz(trash_path, mask_trash)
            if not other_path.exists():
                save_npz(other_path, mask_other)

    def negmask_100x100_paths(self, force=False):
        mask_names = list(self.negmask_learners.keys())
        mask_name_by_paths = self._get_all_100x100_negmask_paths(mask_names)
        if self._all_negmasks_exist(mask_name_by_paths, force):
            return mask_name_by_paths
        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_mask_path())
        all_masks = self._gen_all_negmasks(image, mask)
        self._persist_negmasks(all_masks, mask_name_by_paths)
        return mask_name_by_paths

    def negmask_50x50_paths(self, force=False):
        mask_names = list(self.negmask_learners.keys())
        mask_name_by_paths = self._get_all_50x50_negmask_paths(mask_names)
        if self._all_negmasks_exist(mask_name_by_paths, force):
            return mask_name_by_paths
        image = DiskImage.load(self.image_path)
        mask = DiskBooleanMask.load(self.trimmed_50x50_mask_path())
        all_masks = self._gen_all_negmasks(image, mask)
        self._persist_negmasks(all_masks, mask_name_by_paths)
        return mask_name_by_paths

    def trash_probs_path(self, force=False):
        """Return path to trash probabilities, generate if missing"""
        path = self.d / "trash_probs.npz"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.generate_negmask_probs()
        return path

    def other_probs_path(self, force=False):
        """Return path to other probabilities, generate if missing"""
        path = self.d / "other_probs.npz"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            # Ensure dependencies exist first
            self.generate_negmask_probs()
        return path

    def flower_pos_probs_path(self, force=False):
        """Return path to flower positive probabilities, generate if missing"""
        path = self.d / "flower_pos_probs.npz"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self._generate_flower_probs()
        return path

    def flower_neg_probs_path(self, force=False):
        """Return path to flower negative probabilities, generate if missing"""
        path = self.d / "flower_neg_probs.npz"
        if path.exists() and force:
            path.unlink()
        if not path.exists():
            self._generate_flower_probs()
        return path

    def final_mask_100x100(self):
        mask = DiskBooleanMask.load(self.trimmed_mask_path())
        trash = load_npz(self.trash_probs_path())
        other = load_npz(self.other_probs_path())

        return mask.astype(bool) & (trash > other)

    def final_mask_50x50(self):
        mask = DiskBooleanMask.load(self.trimmed_50x50_mask_path())
        trash = load_npz(self.trash_50x50_probs_path())
        other = load_npz(self.other_50x50_probs_path())

        return mask.astype(bool) & (trash > other)

    def final_mask(self):
        return self.final_mask_100x100() | self.final_mask_50x50()
