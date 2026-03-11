"""
S3 Pipeline Utilities for processing tar files through data processing stages.
"""

import os
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Callable, Optional

from cloudpathlib import S3Path


def process_s3_pipeline_stage(
    s3_root: str,
    source_subdir: str,
    target_subdir: str, 
    processing_function: Callable[[Path, Path], None],
    skip_existing: bool = True
) -> None:
    """
    Process tar files from source subdirectory, apply processing function, and upload to target subdirectory.
    
    Args:
        s3_root: S3 root path (e.g., 's3://bucket/data/')
        source_subdir: Source subdirectory name (e.g., 'images')
        target_subdir: Target subdirectory name (e.g., 'pass_1') 
        processing_function: Function that takes (source_dir, target_dir) and processes files
        skip_existing: Whether to skip processing if target tar already exists
    """
    s3_root_path = S3Path(s3_root)
    source_path = s3_root_path / source_subdir
    target_path = s3_root_path / target_subdir
    
    # Ensure target directory exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of tar files in source directory
    tar_files = list(source_path.glob("*.tar"))
    
    print(f"Found {len(tar_files)} tar files in {source_path}")
    
    for tar_file in tar_files:
        target_tar = target_path / tar_file.name
        
        # Skip if target already exists
        if skip_existing and target_tar.exists():
            print(f"Skipping {tar_file.name} - already exists in {target_subdir}")
            continue
            
        print(f"Processing {tar_file.name}")
        
        # Create temporary directories for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extract_dir = temp_path / "extracted"
            output_dir = temp_path / "output"
            
            # Download and extract tar file
            local_tar = temp_path / tar_file.name
            tar_file.download_to(local_tar)
            
            print(f"  Extracting {tar_file.name}")
            with tarfile.open(local_tar, 'r') as tar:
                tar.extractall(extract_dir)
            
            # Find the root directory inside the extracted tar
            # (tar when expanded would have a single root directory)
            extracted_contents = list(extract_dir.iterdir())
            if len(extracted_contents) != 1 or not extracted_contents[0].is_dir():
                raise ValueError(f"Expected single root directory in {tar_file.name}, found: {[p.name for p in extracted_contents]}")
            
            root_dir = extracted_contents[0]
            
            # Create output directory with same structure
            output_root = output_dir / root_dir.name
            output_root.mkdir(parents=True)
            
            # Apply processing function
            print(f"  Applying processing function to {tar_file.name}")
            processing_function(root_dir, output_root)
            
            # Create new tar file
            output_tar = temp_path / tar_file.name
            print(f"  Creating tar file for {tar_file.name}")
            with tarfile.open(output_tar, 'w') as tar:
                # Add the root directory to the tar
                tar.add(output_root, arcname=root_dir.name)
            
            # Upload to S3
            print(f"  Uploading {tar_file.name} to {target_subdir}")
            target_tar.upload_from(output_tar)
            
        print(f"Completed processing {tar_file.name}")
    
    print(f"Pipeline stage complete: {source_subdir} -> {target_subdir}")


def copy_and_process_images(
    source_dir: Path,
    target_dir: Path,
    processing_function: Optional[Callable[[Path, Path], None]] = None
) -> None:
    """
    Helper function to copy directory structure and optionally apply processing.
    
    Args:
        source_dir: Source directory containing subdirectories with images
        target_dir: Target directory to copy structure to
        processing_function: Optional function to apply to each image directory
    """
    # Copy the directory structure
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            target_subdir = target_dir / subdir.name
            target_subdir.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from source to target
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, target_subdir / file_path.name)
            
            # Apply additional processing if provided
            if processing_function:
                processing_function(subdir, target_subdir)