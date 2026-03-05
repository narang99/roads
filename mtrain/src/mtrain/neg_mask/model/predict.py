import torch
import numpy as np
from .dataset import MaskInferenceDataset
from fastai.vision.all import default_device
from mtrain.neg_mask.crops import get_crops_for_image


def run_inference(learn, img_mask_pairs, device=None) -> torch.Tensor:
    """Return softmax probabilities for all classes, shape [N, C]."""
    from torch.utils.data import DataLoader as TorchDataLoader

    if device is None:
        device = default_device()

    ds = MaskInferenceDataset(img_mask_pairs)
    dl = TorchDataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    learn.model.eval()
    learn.model.to(device)

    all_probs = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            probs = learn.model(x).softmax(dim=1)
            all_probs.append(probs.cpu())

    return torch.cat(all_probs)


def predict_trash(learn, img_mask_pairs, trash_pred_idx=0, threshold=0.25, device=None):
    all_probs = run_inference(learn, img_mask_pairs, device=device)
    trash_probs = all_probs[:, trash_pred_idx]
    predicted_trash = trash_probs >= threshold
    return predicted_trash, trash_probs


def predict_and_reconstruct_mask(
    learn,
    image: np.ndarray,
    mask: np.ndarray,
    trash_pred_idx=0,
    bbox_pad=20,
    crop_pad=220,
):
    bboxes, imgs, masks = [], [], []
    for bbox, crop_img, crop_mask in get_crops_for_image(
        image, mask, bbox_pad, crop_pad
    ):
        bboxes.append(bbox)
        imgs.append(crop_img)
        masks.append(crop_mask)
    if not bboxes:
        return mask.copy().astype(np.float32)

    _, trash_probs = predict_trash(
        learn, list(zip(imgs, masks)), trash_pred_idx=trash_pred_idx, threshold=0
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
