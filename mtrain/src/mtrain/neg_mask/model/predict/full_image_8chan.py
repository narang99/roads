import numpy as np
from mtrain.neg_mask.ipywidgets.bbox_processing import get_region_crops

def predict_and_return_probs(image, mask, learner, crop_pad):
    """Run inference on all bounding boxes and create probability masks."""
    bboxes = list(get_region_crops(image, mask))
    mask = mask.astype(bool)
    if not bboxes or learner is None:
        h, w = mask.shape
        trash_prob_mask = np.zeros((h, w), dtype=np.float32)
        other_prob_mask = np.zeros((h, w), dtype=np.float32)
        return trash_prob_mask, other_prob_mask

    from mtrain.neg_mask.model.predict.predict_8ch import run_inference

    # Prepare crop data for batch inference
    crop_data_list = [(image, mask, bbox) for bbox in bboxes]

    # Run batch inference
    all_probs = run_inference(learner, crop_data_list, crop_pad)  # Shape: [N, C]
    return reconstruct_probability_masks(mask, all_probs.numpy(), bboxes, 0, 1)


def reconstruct_probability_masks(mask, all_probs, bboxes, label_other, label_trash):
    """Map bbox predictions back to full image coordinates."""
    h, w = mask.shape
    trash_prob_mask = np.zeros((h, w), dtype=np.float32)
    other_prob_mask = np.zeros((h, w), dtype=np.float32)

    for i, bbox in enumerate(bboxes):
        # Get probabilities for this bbox
        other_prob = all_probs[i, label_other].item()
        trash_prob = all_probs[i, label_trash].item()

        # Apply to bbox region in full image
        bbox_mask = mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
        other_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = (
            other_prob
        )
        trash_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = (
            trash_prob
        )

    return trash_prob_mask, other_prob_mask

def get_thrash_mask_above_threshold_and_other(
    mask, trash_prob_mask, other_prob_mask, trash_thres
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlay showing the predicted class (highest probability) for each pixel."""
    h, w = mask.shape

    # Find pixels where we have predictions
    has_prediction = (trash_prob_mask > 0) | (other_prob_mask > 0)
    trash_above_other = (trash_prob_mask > other_prob_mask)
    trash_above_thres = (trash_prob_mask > trash_thres)
    return trash_above_other, trash_above_thres, has_prediction
