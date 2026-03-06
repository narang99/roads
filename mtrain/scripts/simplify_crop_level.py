#!/usr/bin/env python3
"""
Script to convert crop_level dataset format to a simplified format.

Original format:
- image.jpg, mask.png (cropped region)
- img_pair_0.jpg, img_pair_1.jpg, img_pair_2.jpg (pre-computed crops)
- mask_pair_0.png, mask_pair_1.png, mask_pair_2.png (pre-computed masks)
- meta.json (crop origin only)
- source_dir symlink

Simplified format:
- image.jpg (full image with mask reconstructed from crop)
- mask.png (single region mask reconstructed from crop)
- original_mask.png (original full mask from source_dir)
- meta.json (original metadata only)
"""

import json
import shutil
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

from mtrain.neg_mask.leveled_cropping import load_crop_level_sample_from_directory
from mtrain.disk import DiskImage, DiskBooleanMask

def process_sample_directory(sample_dir: Path, output_dir: Path) -> bool:
    """Process a single sample directory."""
    # Load original metadata
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        print(f"Warning: {meta_path} does not exist, skipping")
        return False
        
    original_meta = json.loads(meta_path.read_text())
    
    # Check for source_dir symlink
    source_dir = sample_dir / "source_dir"
    if not source_dir.exists():
        print(f"Warning: {source_dir} does not exist, skipping")
        return False
        
    original_image_path = source_dir / "image.jpg"
    if not original_image_path.exists():
        print(f"Warning: {original_image_path} does not exist, skipping")
        return False
    
    original_mask_path = source_dir / "m2.png"
    if not original_mask_path.exists():
        print(f"Warning: {original_mask_path} does not exist, skipping")
        return False
    
    # Create output directory
    output_sample_dir = output_dir / sample_dir.name
    output_sample_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use load_crop_level_sample_from_directory to get reconstructed full image and mask
        sample = load_crop_level_sample_from_directory(sample_dir)
        
        # Save reconstructed full image
        # DiskImage.save(sample.full_image, output_sample_dir / "image.jpg")
        shutil.copy2(original_image_path, output_sample_dir / "image.jpg")
        DiskBooleanMask.save(sample.full_mask, output_sample_dir / "mask.png")
        # Copy original full mask from source
        shutil.copy2(original_mask_path, output_sample_dir / "original_mask.png")
        
        # Save original metadata
        (output_sample_dir / "meta.json").write_text(
            json.dumps(original_meta, indent=2)
        )
        return True
        
        
    except Exception as e:
        print(f"Error processing {sample_dir.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simplify crop_level dataset format")
    parser.add_argument("input_dir", type=Path, help="Input crop_level directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for simplified format")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each class directory (other, trash, unknown)
    good, bad = [], []
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            print(f"Processing class: {class_dir.name}")
            output_class_dir = output_dir / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each sample directory
            for sample_dir in tqdm(class_dir.iterdir()):
                if sample_dir.is_dir():
                    try:
                        success = process_sample_directory(sample_dir, output_class_dir)
                        if success:
                            good.append(sample_dir)
                        else:
                            bad.append(sample_dir)
                    except Exception as e:
                        print(f"Error processing {sample_dir}: {e}")
    print("DONE")
    print("Success count", len(good))
    print("Failed count", len(bad))
    print("Failures", bad)

if __name__ == "__main__":
    main()