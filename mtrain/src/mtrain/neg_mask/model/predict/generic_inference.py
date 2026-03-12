import numpy as np
from typing import Callable, Tuple, Any
from mtrain.neg_mask.ipywidgets.bbox_processing import get_region_crops


def predict_and_return_probs_generic(
    image: np.ndarray, 
    mask: np.ndarray, 
    predictor_fn: Callable,
    *predictor_args,
    **predictor_kwargs
) -> Tuple[np.ndarray, ...]:
    """
    Generic function to run inference on all bounding boxes and create probability masks.
    
    Args:
        image: Input image array
        mask: Boolean mask defining regions of interest
        predictor_fn: Callable that takes (crop_data_list, *args, **kwargs) and returns predictions
        *predictor_args: Additional positional arguments for predictor_fn
        **predictor_kwargs: Additional keyword arguments for predictor_fn
        
    Returns:
        Tuple of probability masks, one for each class
    """
    bboxes = list(get_region_crops(image, mask))
    mask = mask.astype(bool)
    
    if not bboxes or predictor_fn is None:
        h, w = mask.shape
        # Return zero masks for all classes (assume binary classification by default)
        zero_mask = np.zeros((h, w), dtype=np.float32)
        return zero_mask, zero_mask
    
    # Prepare crop data for batch inference
    crop_data_list = [(image, mask, bbox) for bbox in bboxes]
    
    # Run batch inference using the provided predictor function
    all_probs = predictor_fn(crop_data_list, *predictor_args, **predictor_kwargs)
    
    # Reconstruct probability masks for binary classification (can be extended)
    return reconstruct_probability_masks_binary(mask, all_probs.numpy(), bboxes)


def reconstruct_probability_masks_binary(
    mask: np.ndarray, 
    all_probs: np.ndarray, 
    bboxes: list,
    label_class_0: int = 0,
    label_class_1: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map bbox predictions back to full image coordinates for binary classification.
    
    Args:
        mask: Boolean mask defining regions of interest
        all_probs: Prediction probabilities array with shape [N, C]
        bboxes: List of bounding boxes
        label_class_0: Index for first class (default: 0)
        label_class_1: Index for second class (default: 1)
        
    Returns:
        Tuple of (class_0_prob_mask, class_1_prob_mask)
    """
    h, w = mask.shape
    class_0_prob_mask = np.zeros((h, w), dtype=np.float32)
    class_1_prob_mask = np.zeros((h, w), dtype=np.float32)
    
    for i, bbox in enumerate(bboxes):
        # Get probabilities for this bbox
        class_0_prob = all_probs[i, label_class_0].item()
        class_1_prob = all_probs[i, label_class_1].item()
        
        # Apply to bbox region in full image
        bbox_mask = mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
        class_0_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = class_0_prob
        class_1_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = class_1_prob
    
    return class_0_prob_mask, class_1_prob_mask


def reconstruct_probability_masks_multiclass(
    mask: np.ndarray, 
    all_probs: np.ndarray, 
    bboxes: list
) -> Tuple[np.ndarray, ...]:
    """
    Map bbox predictions back to full image coordinates for multi-class classification.
    
    Args:
        mask: Boolean mask defining regions of interest
        all_probs: Prediction probabilities array with shape [N, C]
        bboxes: List of bounding boxes
        
    Returns:
        Tuple of probability masks, one for each class
    """
    h, w = mask.shape
    num_classes = all_probs.shape[1]
    
    # Create probability masks for each class
    prob_masks = [np.zeros((h, w), dtype=np.float32) for _ in range(num_classes)]
    
    for i, bbox in enumerate(bboxes):
        # Apply probabilities for all classes to bbox region
        bbox_mask = mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
        
        for class_idx in range(num_classes):
            class_prob = all_probs[i, class_idx].item()
            prob_masks[class_idx][bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = class_prob
    
    return tuple(prob_masks)


def get_prediction_mask_above_threshold(
    mask: np.ndarray, 
    prob_masks: Tuple[np.ndarray, ...], 
    class_idx: int,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create masks showing predictions above threshold and comparisons between classes.
    
    Args:
        mask: Original boolean mask
        prob_masks: Tuple of probability masks for each class
        class_idx: Index of the class to threshold
        threshold: Probability threshold
        
    Returns:
        Tuple of (class_above_others, class_above_threshold, has_prediction)
    """
    # Find pixels where we have predictions
    has_prediction = np.zeros(mask.shape, dtype=bool)
    for prob_mask in prob_masks:
        has_prediction |= (prob_mask > 0)
    
    # Check if target class has highest probability
    target_prob_mask = prob_masks[class_idx]
    class_above_others = np.zeros(mask.shape, dtype=bool)
    
    for i, other_prob_mask in enumerate(prob_masks):
        if i != class_idx:
            class_above_others |= (target_prob_mask > other_prob_mask)
    
    # Check if above threshold
    class_above_threshold = (target_prob_mask > threshold)
    
    return class_above_others, class_above_threshold, has_prediction