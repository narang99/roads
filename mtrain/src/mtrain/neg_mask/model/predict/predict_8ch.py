from mtrain.neg_mask.crops import get_crops_for_image
import torch
import numpy as np
from ..crop_dataset_base import create_std_transforms
from mtrain.neg_mask.leveled_cropping import (
    create_crop_level_sample,
    make_crop_level_pairs_v2,
)
from torchvision import tv_tensors
from .common import (
    run_inference_common,
    predict_trash_common,
    predict_class_common,
    predict_and_reconstruct_mask_common,
)


def _prepare_8_channel_tensor(
    image: np.ndarray, mask: np.ndarray, bbox, tight_pad=20, medium_pad=130
) -> torch.Tensor:
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

    eval_tfms = create_std_transforms(medium_pad)
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


def run_inference(learn, crop_data_list, medium_pad, device=None) -> torch.Tensor:
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
        tensor_8ch = _prepare_8_channel_tensor(image, mask, bbox, medium_pad)
        tensors.append(tensor_8ch)

    # Stack into batch
    batch_tensor = torch.stack(tensors)  # Shape: [N, 8, 130, 130]

    return run_inference_common(learn, batch_tensor, device)


def reconstruct_probability_masks(
    image, mask, all_probs, bboxes, label_other=0, label_trash=1
):
    """Map bbox predictions back to full image coordinates."""
    h, w = mask.shape
    trash_prob_mask = np.zeros((h, w), dtype=np.float32)
    other_prob_mask = np.zeros((h, w), dtype=np.float32)

    for i, bbox in enumerate(bboxes):
        # Get probabilities for this bbox
        other_prob = all_probs[i, label_other].item()
        trash_prob = all_probs[i, label_trash].item()

        # Apply to bbox region in full image
        print(bbox, other_prob_mask.shape, trash_prob_mask.shape)
        bbox_mask = mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
        other_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = other_prob
        trash_prob_mask[bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = trash_prob
    return trash_prob_mask, other_prob_mask


def predict_trash(
    learn, crop_data_list, medium_pad, trash_pred_idx=1, threshold=0.25, device=None
):
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
    all_probs = run_inference(
        learn, crop_data_list, device=device, medium_pad=medium_pad
    )
    return predict_trash_common(all_probs, trash_pred_idx, threshold)


def predict_class(learn, crop_data_list, medium_pad, device=None):
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
    all_probs = run_inference(
        learn, crop_data_list, medium_pad=medium_pad, device=device
    )
    return predict_class_common(all_probs)


def predict_and_reconstruct_mask(
    learn,
    image: np.ndarray,
    mask: np.ndarray,
    trash_thres,
    bbox_pad=20,
    crop_pad=220,
    label_other=0,
    label_trash=0,
):
    crop_data = []
    bboxes = []
    for bbox, crop_img, crop_mask in get_crops_for_image(
        image, mask, bbox_pad, crop_pad
    ):
        crop_data.append((image, mask, bbox))  # Pass full image/mask + bbox
        bboxes.append(bbox)

    if not bboxes:
        return mask.copy().astype(np.float32)

    all_probs = run_inference(learn, crop_data, crop_pad).numpy()
    trash_prob_mask, other_prob_mask = reconstruct_probability_masks(
        image, mask, all_probs, bboxes, label_other, label_trash
    )
    trash_above_other, trash_above_thres, has_prediction = (
        get_thrash_mask_above_threshold_and_other(
            mask, trash_prob_mask, other_prob_mask, trash_thres
        )
    )
    return trash_above_other, trash_above_thres, has_prediction


def get_thrash_mask_above_threshold_and_other(
    mask, trash_prob_mask, other_prob_mask, trash_thres
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create overlay showing the predicted class (highest probability) for each pixel."""
    h, w = mask.shape

    # Find pixels where we have predictions
    has_prediction = (trash_prob_mask > 0) | (other_prob_mask > 0)
    trash_above_other = trash_prob_mask >= other_prob_mask
    trash_above_thres = trash_prob_mask > trash_thres
    return trash_above_other, trash_above_thres, has_prediction
