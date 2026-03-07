"""
Predict package for mtrain.neg_mask.model

This package provides prediction functionality for both 8-channel and 12-channel models.

Usage:
    # For 12-channel models (CropLevelDataset)
    from mtrain.neg_mask.model.predict import predict_12ch
    probs = predict_12ch.run_inference(learn, crop_data_list)
    
    # For 8-channel models (CropLevelDataset2Chan)  
    from mtrain.neg_mask.model.predict import predict_8ch
    probs = predict_8ch.run_inference(learn, crop_data_list)
    
    # Common utilities
    from mtrain.neg_mask.model.predict.common import get_trash_mask
"""

# Import submodules for easy access
from . import predict_12ch
from . import predict_8ch
from . import common

# Import commonly used functions at package level
from .common import get_trash_mask

# Expose main prediction functions with clear naming
from .predict_12ch import (
    run_inference as run_inference_12ch,
    predict_trash as predict_trash_12ch,
    predict_class as predict_class_12ch,
    predict_and_reconstruct_mask as predict_and_reconstruct_mask_12ch,
)

from .predict_8ch import (
    run_inference as run_inference_8ch,
    predict_trash as predict_trash_8ch,
    predict_class as predict_class_8ch,
    predict_and_reconstruct_mask as predict_and_reconstruct_mask_8ch,
)

__all__ = [
    # Submodules
    'predict_12ch',
    'predict_8ch', 
    'common',
    
    # Common functions
    'get_trash_mask',
    
    # 12-channel functions
    'run_inference_12ch',
    'predict_trash_12ch',
    'predict_class_12ch',
    'predict_and_reconstruct_mask_12ch',
    
    # 8-channel functions  
    'run_inference_8ch',
    'predict_trash_8ch',
    'predict_class_8ch',
    'predict_and_reconstruct_mask_8ch',
]