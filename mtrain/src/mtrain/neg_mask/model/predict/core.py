import torch
import numpy as np
from fastai.vision.all import default_device
from mtrain.neg_mask.crops import get_region_crops


def run_inference(learn, batch_tensor, device=None) -> torch.Tensor:
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


def reconstruct_probability_masks(image, mask, all_probs, bboxes):
    """Map bbox predictions back to full image coordinates."""
    num_labels = all_probs.shape[1]
    h, w = mask.shape
    res_masks = [np.zeros((h, w), dtype=np.float32) for i in range(num_labels)]
    for i, bbox in enumerate(bboxes):
        for label_idx in range(num_labels):
            current_prob = all_probs[i, label_idx].item()
            bbox_mask = mask[bbox.y : bbox.y2, bbox.x : bbox.x2]
            res_masks[label_idx][bbox.y : bbox.y2, bbox.x : bbox.x2][bbox_mask] = (
                current_prob
            )

    return res_masks


def get_crop_data_list(image, mask):
    bboxes = list(get_region_crops(image, mask))
    crop_data_list = [(image, mask, bbox) for bbox in bboxes]
    return crop_data_list
