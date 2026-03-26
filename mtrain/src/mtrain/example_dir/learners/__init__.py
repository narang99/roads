from mtrain.neg_mask.model.datasets.blur_pad_dl import CropTfmsOutsideBbox
import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
from mtrain.smallnet.unet.predict.strided import single, multiple
from mtrain.neg_mask.model.predict import trash as pred_trash
from .smallnet import get_smallnet_learner as get_raw_smallnet_learner
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

    def predict(self, image, mask):
        return pred_trash.predict_and_return_prob_masks_using_unblurred(
            image,
            mask,
            self.learner,
            self.crop_size,
            bbox_pad=self.bbox_pad,
            mutator=self.mutator,
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


def step_downer(cropped_image, mask, inner_bbox):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down(0.5)
    return tfm.crop