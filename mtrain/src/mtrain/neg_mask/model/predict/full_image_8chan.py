import numpy as np
from .generic_inference import (
    predict_and_return_probs_generic,
    get_prediction_mask_above_threshold
)


def predict_and_return_probs(image, mask, learner, crop_pad):
    """Run inference on all bounding boxes and create probability masks."""
    from mtrain.neg_mask.model.predict.predict_8ch import run_inference
    
    def predictor_fn(crop_data_list, learner, crop_pad):
        return run_inference(learner, crop_data_list, crop_pad)
    
    return predict_and_return_probs_generic(
        image, mask, predictor_fn, learner, crop_pad
    )


def reconstruct_probability_masks(mask, all_probs, bboxes, label_other, label_trash):
    """Map bbox predictions back to full image coordinates."""
    from .generic_inference import reconstruct_probability_masks_binary
    return reconstruct_probability_masks_binary(
        mask, all_probs, bboxes, label_other, label_trash
    )

def get_thrash_mask_above_threshold_and_other(
    mask, trash_prob_mask, other_prob_mask, trash_thres
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlay showing the predicted class (highest probability) for each pixel."""
    prob_masks = (other_prob_mask, trash_prob_mask)  # class 0, class 1
    return get_prediction_mask_above_threshold(mask, prob_masks, 1, trash_thres)
