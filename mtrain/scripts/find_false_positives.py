# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %%
from pathlib import Path
import cv2
import numpy as np
import random
from tqdm import tqdm
from mtrain.utils import DiskBooleanMask
from mtrain.example_dir.core import load_npz
from mtrain.seg.mapillary import cached_predict, Label, get_mask

# %%
# Configuration
OUT_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/inference/false_positives")
CHUNKS_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/inference/delhi/chunks")

# Labels to analyze for false positives
TARGET_LABELS = [
    Label.FENCE,
    Label.WALL, 
    Label.RAIL_TRACK,
    Label.LANE_MARKING_GENERAL,
    Label.LANE_MARKING_CROSSWALK,
    Label.CURB,
    Label.BARRIER,
    Label.GUARD_RAIL,
    Label.WATER
]

# Minimum overlap threshold (pixels) to consider a false positive
MIN_OVERLAP_PIXELS = 10

# %%
def setup_output_dirs():
    """Create output directories for each target label"""
    OUT_DIR.mkdir(exist_ok=True)
    for label in TARGET_LABELS:
        label_dir = OUT_DIR / label.name.lower()
        label_dir.mkdir(exist_ok=True)
        print(f"Created directory: {label_dir}")

# %%
def get_all_inference_dirs():
    """Find all directories with both trash and other predictions"""
    all_dirs = []
    
    for chunk_name in range(66):
        chunk_dir = CHUNKS_DIR / str(chunk_name)
        if not chunk_dir.exists():
            continue
            
        for d in chunk_dir.glob("*"):
            ntp = d / "negmask-trash-md.npz"
            nop = d / "negmask-other-md.npz" 
            img_path = d / "image.jpg"
            mapi = d / "mapi.png"
            mask_md = d / "mask-md.png"
            
            if ntp.exists() and nop.exists() and img_path.exists() and mapi.exists() and mask_md.exists():
                all_dirs.append(d)
    
    print(f"Found {len(all_dirs)} directories with predictions")
    return all_dirs

# %%
def get_trash_mask(directory, label="md"):
    """Get binary trash mask where trash > other"""
    trash_pred = load_npz(directory / f"negmask-trash-{label}.npz")
    other_pred = load_npz(directory / f"negmask-other-{label}.npz")
    return trash_pred > other_pred

# %%
def find_all_false_positives(directories, target_labels, max_samples=None):
    """Find directories where trash mask overlaps with mapillary labels in single pass"""
    # Initialize results dict for each label
    results = {label: [] for label in target_labels}
    
    if max_samples:
        directories = random.sample(directories, min(max_samples, len(directories)))
    
    for directory in tqdm(directories, desc="Processing all labels"):
        try:
            # Load all files once per directory
            mask = DiskBooleanMask.load(directory / "mask-md.png")
            trash_mask = get_trash_mask(directory)
            mapi_pred = DiskBooleanMask.load(directory / "mapi.png")  # Load once
            
            # Apply road mask constraint
            constrained_trash = mask & trash_mask
            
            # Check overlap for each label using the same loaded pred
            for label in target_labels:
                # Get individual label mask from same loaded prediction
                mapi_mask = get_mask(mapi_pred, label)
                if mapi_mask is None:
                    continue
                    
                # Find overlap between trash predictions and mapillary label
                overlap = constrained_trash & mapi_mask
                overlap_pixels = overlap.sum()
                
                if overlap_pixels >= MIN_OVERLAP_PIXELS:
                    results[label].append({
                        'directory': directory,
                        'overlap_pixels': overlap_pixels,
                        'total_trash_pixels': constrained_trash.sum(),
                        'total_label_pixels': mapi_mask.sum()
                    })
        except Exception as e:
            print(f"Error processing {directory}: {e}")
            continue
    
    # Sort results for each label by overlap pixels (most problematic first)
    for label in target_labels:
        results[label].sort(key=lambda x: x['overlap_pixels'], reverse=True)
        print(f"Found {len(results[label])} false positives for {label.name.lower()}")
    
    return results

# %%
def copy_directories(false_positives, target_label: Label):
    """Copy false positive directories"""
    import shutil
    
    label_name = target_label.name.lower()
    output_dir = OUT_DIR / label_name
    
    for fp in false_positives:
        source_dir = fp['directory']
        copy_path = output_dir / source_dir.name
        
        try:
            if copy_path.exists():
                shutil.rmtree(copy_path)
            shutil.copytree(source_dir, copy_path)
        except Exception as e:
            print(f"Error copying directory {copy_path}: {e}")

# %%
def analyze_all_labels(directories, target_labels, max_samples=5000):
    """Analyze false positives for all labels in single pass"""
    print(f"\n=== Analyzing {len(target_labels)} labels in single pass ===")
    
    all_results = find_all_false_positives(directories, target_labels, max_samples)
    
    for label, false_positives in all_results.items():
        print(f"\n--- Processing {label.name} results ---")
        
        if false_positives:
            copy_directories(false_positives, label)

            
            # Print summary stats
            total_overlap = sum(fp['overlap_pixels'] for fp in false_positives)
            avg_overlap = total_overlap / len(false_positives)
            print(f"Total overlap pixels: {total_overlap}")
            print(f"Average overlap per case: {avg_overlap:.1f}")
            print(f"Max overlap: {false_positives[0]['overlap_pixels']} pixels")
        else:
            print("No false positives found")
            
    return all_results

# %%
def main():
    """Main execution function"""
    print("Setting up output directories...")
    setup_output_dirs()
    
    print("Finding all inference directories...")
    all_directories = get_all_inference_dirs()
    
    print(f"Processing {len(TARGET_LABELS)} labels in single pass...")
    results = analyze_all_labels(all_directories, TARGET_LABELS, None)
    
    # Print overall summary
    print("\n=== SUMMARY ===")
    for label, fps in results.items():
        print(f"{label.name}: {len(fps)} false positives")
    
    print(f"\nResults saved in: {OUT_DIR}")

# %%
if __name__ == "__main__":
    main()