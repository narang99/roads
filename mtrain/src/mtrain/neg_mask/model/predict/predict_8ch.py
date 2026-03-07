import torch
import numpy as np
from ..crop_dataset_base import create_std_transforms
from mtrain.neg_mask.leveled_cropping import create_crop_level_sample, make_crop_level_pairs_v2
from torchvision import tv_tensors
from .common import (
    run_inference_common, 
    predict_trash_common, 
    predict_class_common,
    predict_and_reconstruct_mask_common
)


def _prepare_8_channel_tensor(image: np.ndarray, mask: np.ndarray, bbox, tight_pad=20, medium_pad=130) -> torch.Tensor:
    """
    Convert single image/mask/bbox to 8-channel tensor format.
    
    Returns tensor of shape [8, 130, 130] where:
    - Channels 0-2: tight crop RGB
    - Channels 3-5: medium crop RGB  
    - Channels 6-7: corresponding masks (tight, medium)
    """
    # Create crop level sample and generate pairs
    sample = create_crop_level_sample(image, mask, bbox)
    pairs_obj = make_crop_level_pairs_v2(sample, tight_pad, medium_pad)
    
    # Convert to tensor format and take only first 2 pairs (tight, medium)
    tensor_pairs = [
        (
            tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
            tv_tensors.Mask(torch.from_numpy(msk.reshape(1, *msk.shape))),
        )
        for img, msk in pairs_obj.pairs[:2]  # Only tight and medium crops
    ]
    
    eval_tfms = create_std_transforms()
    # Apply transforms and stack
    t_images, t_masks = [], []
    for img_tensor, mask_tensor in tensor_pairs:
        t_img, t_mask = eval_tfms((img_tensor, mask_tensor))
        t_images.append(t_img)
        t_masks.append(t_mask)
    
    # Combine: [2 images] + [2 masks] = 8 channels
    t_res = []
    t_res.extend(t_images)
    t_res.extend(t_masks)
    
    combined = torch.cat(t_res, dim=0)  # Shape: [8, 130, 130]
    return combined


def run_inference(learn, crop_data_list, device=None) -> torch.Tensor:
    """
    Run inference on 8-channel crop data.
    
    Args:
        learn: FastAI learner with 8-channel model
        crop_data_list: List of (image, mask, bbox) tuples
        device: Device to run inference on
        
    Returns:
        Softmax probabilities for all classes, shape [N, C]
    """
    # Convert crop data to 8-channel tensors
    tensors = []
    for image, mask, bbox in crop_data_list:
        tensor_8ch = _prepare_8_channel_tensor(image, mask, bbox)
        tensors.append(tensor_8ch)
    
    # Stack into batch
    batch_tensor = torch.stack(tensors)  # Shape: [N, 8, 130, 130]
    
    return run_inference_common(learn, batch_tensor, device)


def predict_trash(learn, crop_data_list, trash_pred_idx=1, threshold=0.25, device=None):
    """
    Predict trash for crop data using 8-channel model.
    
    Args:
        learn: FastAI learner with 8-channel model
        crop_data_list: List of (image, mask, bbox) tuples
        trash_pred_idx: Class index for trash (1 for new model)
        threshold: Probability threshold for trash classification
        device: Device to run inference on
        
    Returns:
        predicted_trash: Boolean tensor of trash predictions
        trash_probs: Probability scores for trash class
    """
    all_probs = run_inference(learn, crop_data_list, device=device)
    return predict_trash_common(all_probs, trash_pred_idx, threshold)


def predict_class(learn, crop_data_list, device=None):
    """
    Predict class for crop data using 8-channel model (argmax, no threshold).
    
    Args:
        learn: FastAI learner with 8-channel model
        crop_data_list: List of (image, mask, bbox) tuples
        device: Device to run inference on
        
    Returns:
        predicted_classes: Tensor of predicted class indices
        all_probs: All class probabilities, shape [N, C]
    """
    all_probs = run_inference(learn, crop_data_list, device=device)
    return predict_class_common(all_probs)


def predict_and_reconstruct_mask(
    learn,
    image: np.ndarray,
    mask: np.ndarray,
    trash_pred_idx=1,
    bbox_pad=20,
    crop_pad=220,
):
    """
    Predict and reconstruct trash mask using 8-channel model.
    
    Args:
        learn: FastAI learner with 8-channel model
        image: Full RGB image
        mask: Binary mask with connected components
        trash_pred_idx: Class index for trash (1 for new model)
        bbox_pad: Padding for bbox detection
        crop_pad: Padding for crop generation (not used in 8-channel)
        
    Returns:
        reconstructed: Float mask with trash probabilities
    """
    def predict_trash_fn(crop_data_list, trash_pred_idx, threshold):
        return predict_trash(learn, crop_data_list, trash_pred_idx, threshold)
    
    return predict_and_reconstruct_mask_common(
        predict_trash_fn, image, mask, trash_pred_idx, bbox_pad, crop_pad
    )