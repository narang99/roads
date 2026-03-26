from prompt_toolkit.key_binding.bindings.completion import display_completions_like_readline
from mtrain.example_dir.learners import SmallnetLearner, NegmaskLearner
from tqdm import tqdm
from itertools import batched
from mtrain.example_dir.trim import get_mask_with_area_in_range
from mtrain.example_dir.mapi_cons import MAPI_LABELS_TO_EXCLUDE, ELEV_LABELS_TO_EXCLUDE
from pathlib import Path
import numpy as np
import cv2
from mtrain.disk import DiskImage, DiskBooleanMask
from mtrain.seg import mapillary as mapi, elevated_vegetation as elev


def _unlinked_if_force(p, force):
    p = Path(p)
    if p.exists() and force:
        p.unlink()
    return p


def load_npz(path):
    with np.load(path) as compressed:
        return compressed["data"]


def save_npz(path, data):
    np.savez_compressed(path, data=data)



def LD(retval):
    if isinstance(retval, Path):
        if retval.name == "image.jpg":
            return ExampleDir.load_and_resize_image(retval)
        elif ("mask" in retval.name or "m2" in retval.name) and retval.suffix == '.png':
            return DiskBooleanMask.load(retval)
        elif retval.name in ["mapi.png", "elev.png"]:
            return DiskBooleanMask.load(retval)
        else:
            raise Exception(f"unknown path for loading {retval}")
    elif isinstance(retval, tuple) and len(retval) == 2 and isinstance(retval[0], Path) and isinstance(retval[1], Path):
        return load_npz(retval[0]), load_npz(retval[1])
    else:
        raise Exception(f"unknown type for loading {retval}")



class ExampleDir:
    """Abstraction of the single directory in inference for the full inference pipeline

    We have stages in inference, this class provides all stages for each directory
    The directory is called "example" here
    This class maintains the code for running inference and persisting it at correct places

    There is one problem though, it does not provide a lot of testing without deleting older results
    I will need to add a simpler API for simply getting the inference along with the original one to test

    There are simple flows. For each size, we run smallnet path, trimmed path, and negmask path
    each flow just has a label (sm/md/test/etc)
    each flow has an associated learner combination
    """

    def __init__(
        self,
        direc: Path | str,
        label_by_smallnet_learners: dict[str, SmallnetLearner],
        label_by_negmask_learners: dict[str, NegmaskLearner],
        flower_learner=None,
        mapillary_segformer=None,
        elev_segformer=None,
    ):
        self.d = Path(direc)
        if not (self.d / "image.jpg").exists():
            raise Exception(f"image {self.d / 'image.jpg'} does not exist")

        self.label_by_smallnet = label_by_smallnet_learners
        self.label_by_negmask = label_by_negmask_learners
        self.mapillary_segformer = mapillary_segformer
        self.elev_segformer = elev_segformer

    @property
    def image_path(self):
        return self.d / "image.jpg"

    @classmethod
    def load_and_resize_image(cls, image_path):
        """Load and resize image if needed (from gen_preds.py logic)"""
        img_arr = DiskImage.load(image_path)
        if img_arr.shape[0] > 1024 or img_arr.shape[1] > 1024:
            img_arr = cv2.resize(img_arr, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        return img_arr

    def _get_smallnet_mask_path(self, label: str):
        return self.d / self._smallnet_path_name(label)

    @classmethod
    def _smallnet_path_name(cls, label):
        return f"mask-{label}.png"

    def smallnet_mask_path(self, label: str, force=False):
        path = _unlinked_if_force(self._get_smallnet_mask_path(label), force)
        if not path.exists():
            smn = self.label_by_smallnet[label]
            img_arr = self.load_and_resize_image(self.image_path)
            mask = smn.predict(img_arr)
            DiskBooleanMask.save(mask, path)
        return path

    @classmethod
    def batch_predict_smallnet_masks(cls, sml: SmallnetLearner, edirs: list["ExampleDir"], bs=2, force=False) -> None:
        # Filter out edirs whose output paths already exist (unless force=True)
        edirs_to_process = []
        if force:
            # If force=True, remove existing files and process all edirs
            for edir in edirs:
                mask_path = edir._get_smallnet_mask_path(sml.label)
                _unlinked_if_force(mask_path, force)
                edirs_to_process.append(edir)
        else:
            # If force=False, only process edirs where output doesn't exist
            for edir in edirs:
                mask_path = edir._get_smallnet_mask_path(sml.label)
                if not mask_path.exists():
                    edirs_to_process.append(edir)
        
        # If no edirs need processing, return early
        if not edirs_to_process:
            return
        
        # Process the filtered edirs
        masks = []
        batches = list(batched(edirs_to_process, bs))
        for batch in tqdm(batches):
            batch = [cls.load_and_resize_image(edir.image_path) for edir in batch]
            masks.extend(SmallnetLearner.batch_predict(sml, batch))
        
        # Verify correct number of predictions
        if len(masks) != len(edirs_to_process):
            raise Exception(f"mismatch in predicted masks length, masks={len(masks)} edirs_to_process={len(edirs_to_process)}")
        
        # Save predictions for the processed edirs
        for mask, edir in zip(masks, edirs_to_process):
            mask_path = edir._get_smallnet_mask_path(sml.label)
            DiskBooleanMask.save(mask, mask_path)


    @classmethod
    def _get_negmask_pathnames(cls, label):
        return (f"negmask-other-{label}.npz", f"negmask-trash-{label}.npz")

    @classmethod
    def batch_predict_negmask_masks(cls, nml: NegmaskLearner, edirs: list["ExampleDir"], from_smallnet_label: str, bs=256, force=False) -> None:
        """Batch predict negmask probabilities for multiple ExampleDirs"""
        # Filter out edirs whose output paths already exist (unless force=True)
        other_path_name, trash_path_name = cls._get_negmask_pathnames(nml.label)
        edirs_to_process = []
        
        if force:
            # If force=True, remove existing files and process all edirs
            for edir in edirs:
                other_path = edir.d / other_path_name
                trash_path = edir.d / trash_path_name
                _unlinked_if_force(other_path, force)
                _unlinked_if_force(trash_path, force)
                edirs_to_process.append(edir)
        else:
            # If force=False, only process edirs where output doesn't exist
            for edir in edirs:
                other_path = edir.d / other_path_name
                trash_path = edir.d / trash_path_name
                if not other_path.exists() or not trash_path.exists():
                    edirs_to_process.append(edir)
        
        # If no edirs need processing, return early
        if not edirs_to_process:
            return
        
        # Process the filtered edirs
        results = []
        batches = list(batched(edirs_to_process, bs))
        for batch in tqdm(batches):
            images = [cls.load_and_resize_image(edir.image_path) for edir in batch]
            masks = [DiskBooleanMask.load(edir.trimmed_mask_path(from_smallnet_label)) for edir in batch]
            results.extend(NegmaskLearner.batch_predict(nml, images, masks))
        
        # Verify correct number of predictions
        if len(results) != len(edirs_to_process):
            raise Exception(f"mismatch in predicted results length, results={len(results)} edirs_to_process={len(edirs_to_process)}")
        
        # Save predictions for the processed edirs
        for (other_probs, trash_probs), edir in zip(results, edirs_to_process):
            other_path = edir.d / other_path_name
            trash_path = edir.d / trash_path_name
            save_npz(other_path, other_probs)
            save_npz(trash_path, trash_probs)

    def trimmed_mask_path(self, label, force=False):
        prev = self.smallnet_mask_path(label)
        path = _unlinked_if_force(self.d / self._trimmed_mask_path_name(label), force)
        if not path.exists():
            smn = self.label_by_smallnet[label]
            mask = DiskBooleanMask.load(prev)
            trimmed = self._get_trimmed_mask(mask)
            trimmed = get_mask_with_area_in_range(trimmed, smn.area_low, smn.area_high)
            DiskBooleanMask.save(trimmed, path)
        return path

    @classmethod
    def _trimmed_mask_path_name(cls, label):
        return f"m2-{label}.png"

    def get_trash_mask(self, label, from_smallnet_label):
        """Get binary trash mask where trash > other"""
        o, t = self.negmask_paths(label, from_smallnet_label)
        o, t = load_npz(o), load_npz(t)
        return t > o

    def negmask_paths(self, label, from_smallnet_label, force=False):
        """Generate trash/other probability arrays"""

        other_path_name, trash_path_name = self._get_negmask_pathnames(label)

        other_path = _unlinked_if_force(self.d / other_path_name, force)
        trash_path = _unlinked_if_force(self.d / trash_path_name, force)

        if not other_path.exists() or not trash_path.exists():
            nml = self.label_by_negmask[label]
            image = DiskImage.load(self.image_path)
            mask = DiskBooleanMask.load(self.trimmed_mask_path(from_smallnet_label))
            other_probs, trash_probs = nml.predict(image, mask)
            save_npz(other_path, other_probs)
            save_npz(trash_path, trash_probs)
        return other_path, trash_path

    def _generate_mapi_mask(self):
        """Generate mapillary segmentation mask"""
        img = self.load_and_resize_image(self.image_path)
        mapi_pred = mapi.predict_with_array(img)
        DiskBooleanMask.save(mapi_pred.astype(np.uint8), self.d / "mapi.png")
        return mapi_pred

    def _generate_elev_mask(self):
        """Generate elevated vegetation segmentation mask"""
        img = self.load_and_resize_image(self.image_path)
        elev_pred = elev.predict_with_array(img, self.elev_segformer)
        DiskBooleanMask.save(elev_pred.astype(np.uint8), self.d / "elev.png")
        return elev_pred

    def _get_trimmed_mask(self, mask):
        """Trim mask by removing unwanted segments (from gen_preds.py)"""
        elev_pred = DiskBooleanMask.load(self.elev_mask_path())
        mapi_pred = DiskBooleanMask.load(self.mapi_mask_path())

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

    def mapi_mask_path(self, force=False):
        """Return path to mapillary mask, generate if missing"""
        path = _unlinked_if_force(self.d / "mapi.png", force)
        if not path.exists():
            self._generate_mapi_mask()
        return path

    def elev_mask_path(self, force=False):
        """Return path to elevation mask, generate if missing"""
        path = _unlinked_if_force(self.d / "elev.png", force)
        if not path.exists():
            self._generate_elev_mask()
        return path

    def single_label_pipeline(self, label):
        mask = DiskBooleanMask.load(self.trimmed_mask_path(label))
        other_path, trash_path = self.negmask_paths(label, label)
        other, trash = load_npz(other_path), load_npz(trash_path)
        return mask.astype(bool) & (trash > other)



    def load_all_assets(self, smallnet_label, negmask_label):
        res = {}
        try:
            res["image"] = DiskImage.load(self.d / "image.jpg")
        except:
            pass

        try:
            res["mask"] = DiskBooleanMask.load(self.smallnet_mask_path(smallnet_label))
        except:
            pass
        try:
            res["m2"] = DiskBooleanMask.load(self.trimmed_mask_path(smallnet_label))
        except:
            pass
        try:
            o, t = self.negmask_paths(negmask_label, smallnet_label)
            o, t = load_npz(o), load_npz(t)
            res["other"], res["trash"] = o, t
            res["trash_mask"] = t > o
        except:
            pass
        try:
            res["mapi_pred"] = DiskBooleanMask.load(self.mapi_mask_path())
        except:
            pass
        try:
            res["elev_pred"] = DiskBooleanMask.load(self.elev_mask_path())
        except:
            pass
    
        return res



    # def flower_pos_probs_path(self, from_smallnet_label, force=False):
    #     """Return path to flower positive probabilities, generate if missing"""
    #     path = self.d / "flower_pos_probs.npz"
    #     if path.exists() and force:
    #         path.unlink()
    #     if not path.exists():
    #         self._generate_flower_probs(from_smallnet_label)
    #     return path

    # def flower_neg_probs_path(self, from_smallnet_label, force=False):
    #     """Return path to flower negative probabilities, generate if missing"""
    #     path = self.d / "flower_neg_probs.npz"
    #     if path.exists() and force:
    #         path.unlink()
    #     if not path.exists():
    #         self._generate_flower_probs(from_smallnet_label)
    #     return path

    # def _generate_flower_probs(self, from_smallnet_label):
    #     """Generate flower pos/neg probability arrays using 3-channel ResNet"""

    #     image = DiskImage.load(self.image_path)
    #     mask = DiskBooleanMask.load(self.trimmed_mask_path(from_smallnet_label))

    #     flower_neg_probs, flower_pos_probs = pred_flower.predict_and_return_prob_masks(
    #         image, mask, self.flower_learner, FLOWER_CROP_SIZE
    #     )

    #     save_npz(self.d / "flower_pos_probs.npz", flower_pos_probs)
    #     save_npz(self.d / "flower_neg_probs.npz", flower_neg_probs)

    #     return flower_pos_probs, flower_neg_probs



# FLOWER_CROP_SIZE = 60
# @functools.lru_cache(maxsize=1)
# def get_default_flower_learner():
#     """Load default flower learner"""
#     FLOWER_MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/flowers/inat/train/size-100_crop-60/models/with-aug-tfms-iter-15.pkl"
#     return load_learner(FLOWER_MODEL_PATH)
