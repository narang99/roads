import torch
import numpy as np
from fastai.vision.all import default_device
from mtrain.neg_mask.crops import get_crops_for_image


def run_inference_common(learn, batch_tensor, device=None) -> torch.Tensor:
    """
    Common inference logic for any channel configuration.
    
    Args:
        learn: FastAI learner
        batch_tensor: Preprocessed batch tensor of shape [N, C, H, W]
        device: Device to run inference on
        
    Returns:
        Softmax probabilities for all classes, shape [N, C]
    """
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    if device is None:
        device = default_device()
    
    # Create dataset and dataloader
    ds = TensorDataset(batch_tensor)
    dl = TorchDataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    learn.model.eval()
    learn.model.to(device)

    all_probs = []
    with torch.no_grad():
        for (x,) in dl:
            x = x.to(device)
            probs = learn.model(x).softmax(dim=1)
            all_probs.append(probs.cpu())

    return torch.cat(all_probs)


def predict_trash_common(all_probs, trash_pred_idx=1, threshold=0.25):
    """
    Common trash prediction logic.
    
    Args:
        all_probs: Class probabilities tensor [N, C]
        trash_pred_idx: Class index for trash (1 for new model)
        threshold: Probability threshold for trash classification
        
    Returns:
        predicted_trash: Boolean tensor of trash predictions
        trash_probs: Probability scores for trash class
    """
    trash_probs = all_probs[:, trash_pred_idx]
    predicted_trash = trash_probs >= threshold
    return predicted_trash, trash_probs


def predict_class_common(all_probs):
    """
    Common class prediction logic (argmax, no threshold).
    
    Args:
        all_probs: Class probabilities tensor [N, C]
        
    Returns:
        predicted_classes: Tensor of predicted class indices
        all_probs: All class probabilities, shape [N, C]
    """
    predicted_classes = all_probs.argmax(dim=1)
    return predicted_classes, all_probs


def get_trash_mask(reconstructed: np.ndarray, threshold=0.25) -> np.ndarray:
    """
    Convert reconstructed probability mask to trash classification mask.
    
    Args:
        reconstructed: Float mask with trash probabilities
        threshold: Probability threshold for trash classification
        
    Returns:
        Classification mask: 0 (background), 1 (trash), 2 (non-trash)
    """
    result = reconstructed.copy()
    result[(reconstructed > 0) & (reconstructed < threshold)] = 2
    result[reconstructed >= threshold] = 1
    return result


def predict_and_reconstruct_mask_common(
    predict_trash_fn,
    image: np.ndarray,
    mask: np.ndarray,
    trash_pred_idx=1,
    bbox_pad=20,
    crop_pad=220,
):
    """
    Common mask reconstruction logic.
    
    Args:
        predict_trash_fn: Function to predict trash (should accept crop_data_list and return _, trash_probs)
        image: Full RGB image
        mask: Binary mask with connected components
        trash_pred_idx: Class index for trash (1 for new model)
        bbox_pad: Padding for bbox detection
        crop_pad: Padding for crop generation (not used in leveled cropping)
        
    Returns:
        reconstructed: Float mask with trash probabilities
    """
    # Generate bboxes for connected components
    crop_data = []
    bboxes = []
    for bbox, crop_img, crop_mask in get_crops_for_image(image, mask, bbox_pad, crop_pad):
        crop_data.append((image, mask, bbox))  # Pass full image/mask + bbox
        bboxes.append(bbox)
    
    if not bboxes:
        return mask.copy().astype(np.float32)

    _, trash_probs = predict_trash_fn(
        crop_data, trash_pred_idx=trash_pred_idx, threshold=0
    )

    reconstructed = mask.copy().astype(np.float32)
    for bbox, prob in zip(bboxes, trash_probs):
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        region = reconstructed[y : y + h, x : x + w]
        region[region == 1] = prob.item()

    return reconstructed