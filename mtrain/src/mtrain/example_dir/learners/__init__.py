from mtrain.neg_mask.model.datasets.blur_pad_dl import CropTfmsOutsideBbox
from typing import Any, Callable
from dataclasses import dataclass
from mtrain.smallnet.unet.predict.strided import single
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

    @property
    def pathname(self):
        return f"mask-{self.label}.png"

    @property
    def trimmed_pathname(self):
        return f"m2-{self.label}.png"


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

    @property
    def pathnames(self):
        return (f"negmask-other-{self.label}.npz", f"negmask-trash-{self.label}.npz")


def step_downer(cropped_image, mask, inner_bbox):
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down(0.5)
    return tfm.crop