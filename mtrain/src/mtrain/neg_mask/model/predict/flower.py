from mtrain.neg_mask.model.predict.mk_tensors import mk_3_chan
import numpy as np
from mtrain.neg_mask.model.predict import core
from mtrain.neg_mask.ipywidgets.bbox_processing import get_region_crops
import torch


def run_inference(
    learner, crop_data_list, crop_size: int = 60, device=None
) -> torch.Tensor:
    tensors = [
        mk_3_chan(image, mask, bbox, crop_size) for image, mask, bbox in crop_data_list
    ]
    batch_tensor = torch.stack(tensors)  # Shape: [N, 3, crop_size+20, crop_size+20]
    return core.run_inference(learner, batch_tensor, device)


def predict_and_return_prob_masks(
    image, mask, learner, crop_size: int = 60, device=None
):
    bboxes = list(get_region_crops(image, mask))
    if not bboxes:
        zero_mask = np.zeros(mask.shape, dtype=np.float32)
        return zero_mask, zero_mask
    mask = mask.astype(bool)
    crop_data_list = [(image, mask, bbox) for bbox in bboxes]
    probs = run_inference(learner, crop_data_list, crop_size, device)
    prob_masks = core.reconstruct_probability_masks(image, mask, probs, bboxes)

    neg_mask, pos_mask = prob_masks[0], prob_masks[1]
    return neg_mask, pos_mask
