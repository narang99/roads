import torch

# ImageNet mean and standard deviation
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize_imagenet(tensor):
    """
    Denormalizes a PyTorch image tensor normalized with ImageNet statistics.

    Args:
        tensor (torch.Tensor): Normalized image tensor (C, H, W) or (B, C, H, W).

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    # Clone the tensor to avoid modifying the original in-place
    denorm_tensor = tensor.clone()
    
    # Apply the inverse operation: x = (x * std) + mean
    if denorm_tensor.dim() == 4: # Batched input (B, C, H, W)
        denorm_tensor = denorm_tensor * IMAGENET_STD.unsqueeze(0) + IMAGENET_MEAN.unsqueeze(0)
    elif denorm_tensor.dim() == 3: # Single image input (C, H, W)
        denorm_tensor = denorm_tensor * IMAGENET_STD + IMAGENET_MEAN
    else:
        raise ValueError("Input tensor must be of shape (C, H, W) or (B, C, H, W)")
        
    # Clip values to the valid image range [0, 1] as some values might slightly exceed this range
    # due to floating point operations and original image values.
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    return denorm_tensor


def denormalize_4chan_imagenet(tensor):
    image = tensor[:3,:,:]
    denorm_image = denormalize_imagenet(image)
    mask = tensor[3]
    return denorm_image, mask
