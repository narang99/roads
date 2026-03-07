#!/usr/bin/env python3
"""
Script to copy dataset structure replacing symlinked source_dir with actual directory copies.
"""

import os
import shutil
import argparse
import json
from pathlib import Path


def copy_dataset_item(src_dir, dst_dir):
    """Copy a single dataset item, replacing symlinks with actual directories."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination directory
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files (not source_dir)
    for item in src_path.iterdir():
        if item.name == 'source_dir':
            continue
        if item.name not in ["image.jpg", "mask.png"]:
            continue
        
        dst_item = dst_path / item.name
        if item.is_file():
            shutil.copy2(item, dst_item)
            print(f"Copied file: {item.name}")
    
    # Handle source_dir specially
    source_dir = src_path / 'source_dir'
    if source_dir.exists():
        dst_source_dir = dst_path / 'source_dir'
        
        if source_dir.is_symlink():
            # Resolve symlink and copy the actual directory
            target_path = source_dir.readlink()
            if not target_path.is_absolute():
                target_path = source_dir.parent / target_path
            
            if target_path.exists() and target_path.is_dir():
                shutil.copytree(target_path, dst_source_dir)
                print(f"Copied source_dir from: {target_path}")
            else:
                print(f"Warning: Symlink target does not exist: {target_path}")
        else:
            # If it's already a directory, just copy it
            shutil.copytree(source_dir, dst_source_dir)
            print(f"Copied source_dir directory")


def copy_dataset_structure(input_dir, output_dir):
    """Copy entire dataset structure with resolved symlinks."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return False
    
    # Find all dataset items (directories containing image.jpg, mask.png, meta.json)
    copied_count = 0
    
    for item_dir in input_path.rglob("*"):
        if not item_dir.is_dir():
            continue
        
        # Check if this looks like a dataset item
        has_image = (item_dir / "image.jpg").exists()
        has_mask = (item_dir / "mask.png").exists()
        has_meta = (item_dir / "meta.json").exists()
        
        if has_image and has_mask and has_meta:
            # Calculate relative path from input to this item
            rel_path = item_dir.relative_to(input_path)
            dst_item_dir = output_path / rel_path
            
            print(f"\nProcessing: {rel_path}")
            copy_dataset_item(item_dir, dst_item_dir)
            copied_count += 1
    
    print(f"\nCopied {copied_count} dataset items to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Copy dataset structure with resolved symlinks")
    parser.add_argument("input_dir", help="Input dataset directory")
    parser.add_argument("output_dir", help="Output directory for copied dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        input_path = Path(args.input_dir)
        count = 0
        
        for item_dir in input_path.rglob("*"):
            if not item_dir.is_dir():
                continue
            
            has_image = (item_dir / "image.jpg").exists()
            has_mask = (item_dir / "mask.png").exists()
            has_meta = (item_dir / "meta.json").exists()
            
            if has_image and has_mask and has_meta:
                rel_path = item_dir.relative_to(input_path)
                print(f"Would copy: {rel_path}")
                
                source_dir = item_dir / 'source_dir'
                if source_dir.exists() and source_dir.is_symlink():
                    target = source_dir.readlink()
                    if not target.is_absolute():
                        target = source_dir.parent / target
                    print(f"  - Would resolve symlink: source_dir -> {target}")
                
                count += 1
        
        print(f"Found {count} dataset items that would be copied")
    else:
        copy_dataset_structure(args.input_dir, args.output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())