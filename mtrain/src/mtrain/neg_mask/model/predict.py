import torch
import numpy as np
from .dataset import MaskInferenceDataset, MASK_DS_EVAL_TFMS
from fastai.vision.all import default_device
from mtrain.neg_mask.crops import get_crops_for_image
from mtrain.neg_mask.leveled_cropping import create_crop_level_sample, make_crop_level_pairs_v2
from pathlib import Path
from torchvision import tv_tensors


def _prepare_12_channel_tensor(image: np.ndarray, mask: np.ndarray, bbox, tight_pad=20, medium_pad=130) -> torch.Tensor:
    """
    Convert single image/mask/bbox to 12-channel tensor format.
    
    Returns tensor of shape [12, 130, 130] where:
    - Channels 0-2: tight crop RGB
    - Channels 3-5: medium crop RGB  
    - Channels 6-8: full image RGB
    - Channels 9-11: corresponding masks
    """
    # Create crop level sample and generate pairs
    sample = create_crop_level_sample(image, mask, bbox)
    pairs_obj = make_crop_level_pairs_v2(sample, tight_pad, medium_pad)
    
    # Convert to tensor format (matches GenericMaskClassificationDataset.__getitem__)
    tensor_pairs = [
        (
            tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
            tv_tensors.Mask(torch.from_numpy(msk.reshape(1, *msk.shape))),
        )
        for img, msk in pairs_obj.pairs
    ]
    
    # Apply transforms and stack
    t_images, t_masks = [], []
    for img_tensor, mask_tensor in tensor_pairs:
        t_img, t_mask = MASK_DS_EVAL_TFMS((img_tensor, mask_tensor))
        t_images.append(t_img)
        t_masks.append(t_mask)
    
    # Combine: [3 images] + [3 masks] = 12 channels
    t_res = []
    t_res.extend(t_images)
    t_res.extend(t_masks)
    
    combined = torch.cat(t_res, dim=0)  # Shape: [12, 130, 130]
    return combined


def run_inference(learn, crop_data_list, device=None) -> torch.Tensor:
    """
    Run inference on 12-channel crop data.
    
    Args:
        learn: FastAI learner with 12-channel model
        crop_data_list: List of (image, mask, bbox) tuples
        device: Device to run inference on
        
    Returns:
        Softmax probabilities for all classes, shape [N, C]
    """
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

    if device is None:
        device = default_device()

    # Convert crop data to 12-channel tensors
    tensors = []
    for image, mask, bbox in crop_data_list:
        tensor_12ch = _prepare_12_channel_tensor(image, mask, bbox)
        tensors.append(tensor_12ch)
    
    # Stack into batch
    batch_tensor = torch.stack(tensors)  # Shape: [N, 12, 130, 130]
    
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


def predict_trash(learn, crop_data_list, trash_pred_idx=1, threshold=0.25, device=None):
    """
    Predict trash for crop data using 12-channel model.
    
    Args:
        learn: FastAI learner with 12-channel model
        crop_data_list: List of (image, mask, bbox) tuples
        trash_pred_idx: Class index for trash (1 for new model)
        threshold: Probability threshold for trash classification
        device: Device to run inference on
        
    Returns:
        predicted_trash: Boolean tensor of trash predictions
        trash_probs: Probability scores for trash class
    """
    all_probs = run_inference(learn, crop_data_list, device=device)
    trash_probs = all_probs[:, trash_pred_idx]
    predicted_trash = trash_probs >= threshold
    return predicted_trash, trash_probs


def predict_class(learn, crop_data_list, device=None):
    """
    Predict class for crop data using 12-channel model (argmax, no threshold).
    
    Args:
        learn: FastAI learner with 12-channel model
        crop_data_list: List of (image, mask, bbox) tuples
        device: Device to run inference on
        
    Returns:
        predicted_classes: Tensor of predicted class indices
        all_probs: All class probabilities, shape [N, C]
    """
    all_probs = run_inference(learn, crop_data_list, device=device)
    predicted_classes = all_probs.argmax(dim=1)
    return predicted_classes, all_probs


def predict_and_reconstruct_mask(
    learn,
    image: np.ndarray,
    mask: np.ndarray,
    trash_pred_idx=1,
    bbox_pad=20,
    crop_pad=220,
):
    """
    Predict and reconstruct trash mask using 12-channel model.
    
    Args:
        learn: FastAI learner with 12-channel model
        image: Full RGB image
        mask: Binary mask with connected components
        trash_pred_idx: Class index for trash (1 for new model)
        bbox_pad: Padding for bbox detection
        crop_pad: Padding for crop generation (not used in 12-channel)
        
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

    _, trash_probs = predict_trash(
        learn, crop_data, trash_pred_idx=trash_pred_idx, threshold=0
    )

    reconstructed = mask.copy().astype(np.float32)
    for bbox, prob in zip(bboxes, trash_probs):
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        region = reconstructed[y : y + h, x : x + w]
        region[region == 1] = prob.item()

    return reconstructed


def get_trash_mask(reconstructed: np.ndarray, threshold=0.25) -> np.ndarray:
    result = reconstructed.copy()
    result[(reconstructed > 0) & (reconstructed < threshold)] = 2
    result[reconstructed >= threshold] = 1
    return result
