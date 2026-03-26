# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
%load_ext autoreload
%autoreload 2
%matplotlib inline

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import torch
from mtrain.neg_mask.model.datasets.blur_pad_dl import random_tfm, BlurPadDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from mtrain.utils import show, mkdir, DiskImage, DiskBooleanMask
from pytorch_grad_cam import (
    GradCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from mtrain.neg_mask.model.show import (
    get_preds_for_ds,
    show_classification_report,
    show_confusion_matrix,
    show_confusion_matrix_using_preds,
)
from mtrain.neg_mask.model.datasets.blur_pad_dl import CropTfmsOutsideBbox
from functools import partial
from sklearn.model_selection import train_test_split
from fastai.basics import DataLoaders, default_device
from mtrain.denorm import denormalize_imagenet, denormalize_4chan_imagenet
from mtrain.utils import show, it_chain
from fastai.callback.all import ProgressCallback
from fastai.basics import F1Score, Precision, Recall, CrossEntropyLossFlat
from fastai.vision.all import vision_learner, xresnet18

# %%
CLEAN_PATH = Path(
    "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/blurred/clean/train"
)

CLS_WEIGHT = torch.tensor([1.0, 1.0]).float().to("mps")

def get_learner(dls):
    learn = vision_learner(
        dls,
        xresnet18,
        metrics=[F1Score(average="macro"), Precision(), Recall()],
        loss_func=CrossEntropyLossFlat(CLS_WEIGHT),
        n_out=2,
        normalize=False,
        n_in=3,
        pretrained=True,
    )
    learn = learn.remove_cb(ProgressCallback)
    return learn

# %%
def step_down_tfm(cropped_image, mask, inner_bbox, add_noise_chance, ratio):
    add_noise = random.random() < add_noise_chance
    tfm = CropTfmsOutsideBbox(cropped_image, inner_bbox)
    tfm = tfm.step_down(ratio)
    if add_noise:
        tfm = tfm.add_noise(20)
    return tfm.crop

def get_test_dls(ds_path=CLEAN_PATH, crop_size=224):
    """Create a validation-only dataset from all images in CLEAN_PATH"""
    image_paths = list((ds_path / "train").glob("*.jpg"))
    
    st_ed_tfm = partial(step_down_tfm, ratio=0.5)
    st_ed_tfm0 = partial(st_ed_tfm, add_noise_chance=-1)  # no noise
    
    # Create only validation dataset with all images
    test_ds = BlurPadDataset(
        image_paths,
        ds_path / "masks",
        crop_size,
        True,  # is_valid=True
        crop_mutator=st_ed_tfm0,
        bbox_pad=10,
        min_area=35,
        min_bbox_length=3,
        max_area=None,
    )
    
    # Create dummy train dataset for DataLoaders (required by fastai)
    dummy_ds = BlurPadDataset(
        image_paths[:10],  # just a few images for dummy
        ds_path / "masks", 
        crop_size,
        False,
        crop_mutator=st_ed_tfm0,
        bbox_pad=10,
        min_area=35,
        min_bbox_length=3,
        max_area=None,
    )
    
    dls = DataLoaders.from_dsets(
        dummy_ds,
        test_ds,
        device=default_device(),
        num_workers=4,
        bs=16,
        persistent_workers=True,
    )
    return dls

# %%
def load_model(model_path, dls):
    """Load a model from state dict"""
    learner = get_learner(dls)
    state_dict = torch.load(model_path)
    learner.model.load_state_dict(state_dict)
    learner.eval()
    return learner

# %%
def compare_models(old_model_path, new_model_path, test_dls):
    """
    Compare predictions between old and new models on test dataset
    
    Returns:
        dict: Summary of differences and list of differing indices
    """
    # Load models
    old_model = load_model(old_model_path, test_dls)
    new_model = load_model(new_model_path, test_dls)
    
    # Set test data
    old_model.dls = test_dls
    new_model.dls = test_dls
    
    # Get predictions
    print("Getting predictions from old model...")
    old_preds = old_model.get_preds(dl=test_dls.valid, with_decoded=True)
    old_probs, old_targs, old_decoded = old_preds
    
    print("Getting predictions from new model...")
    new_preds = new_model.get_preds(dl=test_dls.valid, with_decoded=True)
    new_probs, new_targs, new_decoded = new_preds
    
    # Find differences
    diff_mask = old_decoded != new_decoded
    diff_indices = torch.where(diff_mask)[0].tolist()
    
    # Calculate summary statistics
    total_samples = len(old_decoded)
    num_different = len(diff_indices)
    diff_percentage = (num_different / total_samples) * 100
    
    # Analyze changes by class
    old_to_new_changes = {}
    for idx in diff_indices:
        old_pred = old_decoded[idx].item()
        new_pred = new_decoded[idx].item()
        key = f"{old_pred}->{new_pred}"
        old_to_new_changes[key] = old_to_new_changes.get(key, 0) + 1
    
    # Print summary
    print(f"\n=== MODEL COMPARISON SUMMARY ===")
    print(f"Total samples: {total_samples}")
    print(f"Different predictions: {num_different}")
    print(f"Percentage different: {diff_percentage:.2f}%")
    print(f"\nPrediction changes:")
    for change, count in old_to_new_changes.items():
        print(f"  {change}: {count} samples")
    
    # Print some example paths that changed
    print(f"\nExample paths with different predictions (showing first 10):")
    image_paths = test_dls.valid_ds.image_paths
    for i, idx in enumerate(diff_indices[:10]):
        old_pred = old_decoded[idx].item()
        new_pred = new_decoded[idx].item()
        path = image_paths[idx]
        print(f"  {path.name}: {old_pred} -> {new_pred}")
    
    if len(diff_indices) > 10:
        print(f"  ... and {len(diff_indices) - 10} more")
    
    return {
        'total_samples': total_samples,
        'num_different': num_different,
        'diff_percentage': diff_percentage,
        'changes': old_to_new_changes,
        'diff_indices': diff_indices,
        'old_decoded': old_decoded,
        'new_decoded': new_decoded,
        'image_paths': image_paths
    }

# %%
# Set up test data
test_dls = get_test_dls()
print(f"Created test dataset with {len(test_dls.valid_ds)} samples")

# %%
# Model paths
OLD_MODEL_PATH = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/models/successive-224/tfm-stepedge_data-withusefultaco_iter-35_arch-xresnet18.pth")
NEW_MODEL_PATH = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/models/successive-224/st_ed_tfm0-final-with-taco-with-wall-iter-6.pth")

# %%
# Run comparison
results = compare_models(OLD_MODEL_PATH, NEW_MODEL_PATH, test_dls)

# %%
# Additional analysis - show some specific examples
def show_different_predictions(results, start_idx=0, num_to_show=5):
    """Show images where predictions differ between models"""
    diff_indices = results['diff_indices']
    old_decoded = results['old_decoded']
    new_decoded = results['new_decoded']
    image_paths = results['image_paths']
    
    end_idx = min(start_idx + num_to_show, len(diff_indices))
    
    for i in range(start_idx, end_idx):
        idx = diff_indices[i]
        old_pred = old_decoded[idx].item()
        new_pred = new_decoded[idx].item()
        path = image_paths[idx]
        
        print(f"\nSample {i+1}: {path.name}")
        print(f"Old model: {old_pred}, New model: {new_pred}")
        
        # Load and show image
        tens, targ = test_dls.valid_ds[idx]
        from mtrain.denorm import denormalize_imagenet
        img = denormalize_imagenet(tens).permute([1, 2, 0]).numpy()
        
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.title(f"{path.name}\nOld: {old_pred} -> New: {new_pred}")
        plt.axis('off')
        plt.show()

# %%
# Show some examples of different predictions
show_different_predictions(results, 0, 3)