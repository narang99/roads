import numpy as np
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from mtrain.neg_mask.leveled_cropping import (
    create_crop_level_sample,
    make_crop_level_pairs_v2,
)
from mtrain.neg_mask.model.datasets.crop_dataset_base import create_std_transforms


def mk_8_chan(
    image: np.ndarray, mask: np.ndarray, bbox, tight_pad=20, medium_pad=130
) -> torch.Tensor:
    """
    Convert single image/mask/bbox to 8-channel tensor format.

    Returns tensor of shape [8, 130, 130] where:
    - Channels 0-2: tight crop RGB
    - Channels 3-5: medium crop RGB
    - Channels 6-7: corresponding masks (tight, medium)
    """
    tensor_pairs = _get_tensor_pairs(image, mask, bbox, tight_pad, medium_pad)
    t_images, t_masks = _get_transformed(tensor_pairs, medium_pad)
    t_res = t_images + t_masks
    combined = torch.cat(t_res, dim=0)  # Shape: [8, 130, 130]
    return combined


def mk_3_chan(
    image: np.ndarray, mask: np.ndarray, bbox, tight_pad=20, medium_pad=130
) -> torch.Tensor:
    """
    Convert single image/mask/bbox to 3-channel tensor format.

    Returns tensor of shape [3, medium_pad, medium_pad] where:
    - Channels 0-2: tight crop RGB
    """
    tensor_pairs = _get_tensor_pairs(image, mask, bbox, tight_pad, medium_pad)
    t_images, _ = _get_transformed(tensor_pairs[:1], medium_pad)
    return t_images[0]


def _get_tensor_pairs(image, mask, bbox, tight_pad, medium_pad):
    sample = create_crop_level_sample(image, mask, bbox)
    pairs_obj = make_crop_level_pairs_v2(sample, tight_pad, medium_pad)

    # Convert to tensor format and take only first 2 pairs (tight, medium)
    return [
        (
            tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
            tv_tensors.Mask(torch.from_numpy(msk.reshape(1, *msk.shape))),
        )
        for img, msk in pairs_obj.pairs[:2]  # Only tight and medium crops
    ]


def _get_transformed(tensor_pairs, medium_pad):
    # Apply transforms and stack
    eval_tfms = create_std_transforms(medium_pad)
    t_images, t_masks = [], []
    for img_tensor, mask_tensor in tensor_pairs:
        t_img, t_mask = eval_tfms((img_tensor, mask_tensor))
        t_images.append(t_img)
        t_masks.append(t_mask)
    return t_images, t_masks
