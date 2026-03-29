from mtrain.neg_mask.crops import get_largest_bbox
from mtrain.neg_mask.model.datasets.foviate_shrink import get_foviated_image_and_mask
from mtrain.neg_mask.model.datasets.blur_pad_dl import CropTfmsOutsideBbox, BlurPadInferDataset
import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
from mtrain.smallnet.unet.predict.strided import single, multiple
from mtrain.neg_mask.model.predict import trash as pred_trash
from .smallnet import get_smallnet_learner as get_raw_smallnet_learner
from torch.utils.data import Dataset
from .negmask import get_negmask_learner as get_raw_negmask_learner


@dataclass
class SmallnetLearner:
    label: str
    learner: Any
    bs: int
    tile_size: int
    strides: list[int]
    area_low: int | None
    area_high: int | None

    def predict(self, image):
        return single.strided_predict_unet_only_mask(
            image, self.tile_size, self.learner, self.strides, self.bs
        )

    @classmethod
    def batch_predict(cls, sml: "SmallnetLearner", batch: list[np.ndarray]):
        return multiple.strided_predict_unet_only_mask(batch, sml.tile_size, sml.learner, sml.strides, sml.bs)

@dataclass
class NegmaskLearner:
    label: str
    learner: Any
    bs: int
    crop_size: int
    bbox_pad: int
    mutator: Callable
    valid_tfms_crop_size: int | None = None
    dataset_class: Dataset | None = None

    def predict(self, image, mask):
        dataset_class = self.dataset_class
        if self.dataset_class is None:
            dataset_class = BlurPadInferDataset
        return pred_trash.predict_and_return_prob_masks_using_unblurred(
            image,
            mask,
            self.learner,
            self.crop_size,
            bbox_pad=self.bbox_pad,
            mutator=self.mutator,
            valid_tfms_crop_size=self.valid_tfms_crop_size,
            dataset_class=dataset_class,
        )

    @classmethod
    def batch_predict(cls, nml: "NegmaskLearner", images: list[np.ndarray], masks: list[np.ndarray]):
        return pred_trash.batch_predict_and_return_prob_masks(
            images,
            masks,
            nml.learner,
            nml.crop_size,
            bbox_pad=nml.bbox_pad,
            mutator=nml.mutator,
            bs=nml.bs,
        )


@dataclass
class EnsembledNegmaskLearner:
    label: str
    learners: list[NegmaskLearner]

    def predict(self, image, mask):
        other_probs, trash_probs = [], []
        for learner in self.learners:
            o, t = learner.predict(image, mask)
            other_probs.append(o)
            trash_probs.append(t)
        
        mean_other_probs = np.mean(other_probs, axis=0)
        mean_trash_probs = np.mean(trash_probs, axis=0)

        return mean_other_probs, mean_trash_probs

def step_downer(cropped_image, mask, inner_bbox):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down(0.5)
    return tfm.crop, mask


# def foveate_shrink(cropped_image, mask, inner_bbox, full_image_size, crop_size):
#     _, (re_img, re_mask), inner_bbox = get_foviated_image_and_mask(cropped_image, mask, inner_bbox, full_image_size, crop_size, 0)
#     return re_img, re_mask, inner_bbox


def foveate_shrink_and_step_down(cropped_image, mask, inner_bbox):
    # we already get padded box, no need to pad more
    _, (cropped_image, mask), _, bbox = get_foviated_image_and_mask(cropped_image, mask, inner_bbox, 1024, 224, 0)
    # cropped_image, mask, bbox = foveate_shrink(cropped_image, mask, inner_bbox, 1024, 224)
    return step_downer(cropped_image, mask, bbox)