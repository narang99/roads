# all vibe coded
import cv2
import numpy as np

def calculate_perspective_scale_factor(orig_y, new_y, orig_horizon_y, new_horizon_y, orig_height, new_height):
    """
    Calculate perspective scale factor based on normalized distance ratios from horizon.
    
    orig_y: y-coordinate in original image
    new_y: y-coordinate in new image
    orig_horizon_y: horizon y-coordinate in original image
    new_horizon_y: horizon y-coordinate in new image
    orig_height: height of original image
    new_height: height of new image
    
    Returns: scale factor
    """
    # Calculate normalized distances from horizon (as ratio of image height)
    orig_dist_ratio = abs(orig_y - orig_horizon_y) / orig_height
    new_dist_ratio = abs(new_y - new_horizon_y) / new_height
    
    # Avoid division by zero
    if orig_dist_ratio < 0.001:  # Very small threshold for normalized values
        orig_dist_ratio = 0.001
    if new_dist_ratio < 0.001:
        new_dist_ratio = 0.001
    
    # Scale factor based on relative distance ratios
    scale_factor = new_dist_ratio / orig_dist_ratio
    
    # Clamp to reasonable range
    scale_factor = np.clip(scale_factor, 0.1, 3.0)
    
    return scale_factor


def scale_fragment_with_mask(fragment, mask, scale_factor):
    """
    Scale a fragment and its mask by the given factor.
    
    fragment: BGR image fragment
    mask: boolean mask for the fragment
    scale_factor: scaling factor
    
    Returns: scaled_fragment, scaled_mask
    """
    if abs(scale_factor - 1.0) < 0.01:
        # No meaningful scaling needed
        return fragment, mask
    
    # Calculate new dimensions
    old_h, old_w = fragment.shape[:2]
    new_w = int(old_w * scale_factor)
    new_h = int(old_h * scale_factor)
    
    if new_w <= 0 or new_h <= 0:
        # Return minimal fragment if scaling results in zero size
        return np.zeros((1, 1, 3), dtype=fragment.dtype), np.zeros((1, 1), dtype=bool)
    
    # Scale fragment
    scaled_fragment = cv2.resize(fragment, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Scale mask
    mask_uint8 = mask.astype(np.uint8) * 255
    scaled_mask_uint8 = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scaled_mask = scaled_mask_uint8 > 127
    
    return scaled_fragment, scaled_mask

