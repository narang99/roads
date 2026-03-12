from pathlib import Path
from tqdm import tqdm
from typing import List, Union
from .core import ExampleDir, get_default_smallnet_learner, get_default_negmask_learner


def run_bulk_inference(
    directories: list[Path | str | ExampleDir],
    smallnet_learner=None,
    negmask_learner=None,
    mapillary_segformer=None,
    elev_segformer=None,
    verbose=True
):
    """
    Run bulk inference on a list of directories in passes.
    
    Args:
        directories: List of directory paths that contain image.jpg files
        smallnet_learner: Optional smallnet learner, defaults to get_default_smallnet_learner()
        negmask_learner: Optional negmask learner, defaults to get_default_negmask_learner()
        mapillary_segformer: Optional mapillary segformer
        elev_segformer: Optional elevation segformer
        verbose: Whether to show progress bars and stage information
    
    Returns:
        List[ExampleDir]: List of ExampleDir instances for further processing
    """
    # Load models once if not provided
    if smallnet_learner is None:
        if verbose:
            print("Loading default smallnet learner...")
        smallnet_learner = get_default_smallnet_learner()
    
    if negmask_learner is None:
        if verbose:
            print("Loading default negmask learner...")
        negmask_learner = get_default_negmask_learner()
    
    # Create ExampleDir instances
    if verbose:
        print("Setting up ExampleDir instances...")
    example_dirs = []
    for directory in directories:
        try:
            if isinstance(directory, ExampleDir):
                example_dir = directory
            else:
                example_dir = ExampleDir(
                    directory,
                    smallnet_learner=smallnet_learner,
                    negmask_learner=negmask_learner,
                    mapillary_segformer=mapillary_segformer,
                    elev_segformer=elev_segformer
                )
            example_dirs.append(example_dir)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not create ExampleDir for {directory}: {e}")
    
    # Run processing passes
    generate_smallnet_masks(example_dirs, verbose)
    generate_segmentation_masks(example_dirs, verbose)
    generate_trimmed_masks(example_dirs, verbose)
    generate_negmask_probs(example_dirs, verbose)
    
    if verbose:
        print(f"Bulk inference complete for {len(example_dirs)} directories")
    
    return example_dirs


def generate_smallnet_masks(example_dirs: List[ExampleDir], verbose: bool):
    """Pass 1: Generate smallnet masks for all images"""
    if verbose:
        print("STAGE: Pass 1 - Smallnet masks")
    
    iterator = tqdm(example_dirs) if verbose else example_dirs
    for example_dir in iterator:
        try:
            example_dir.smallnet_mask_path()  # This generates the mask if missing
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate smallnet mask for {example_dir.d}: {e}")


def generate_segmentation_masks(example_dirs: List[ExampleDir], verbose: bool):
    """Pass 2: Generate segmentation masks (mapillary and elevation)"""
    if verbose:
        print("STAGE: Pass 2 - Segmentation masks")
    
    iterator = tqdm(example_dirs) if verbose else example_dirs
    for example_dir in iterator:
        try:
            example_dir.mapi_mask_path()  # This generates mapi mask if missing
            example_dir.elev_mask_path()  # This generates elev mask if missing
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate segmentation masks for {example_dir.d}: {e}")


def generate_trimmed_masks(example_dirs: List[ExampleDir], verbose: bool):
    """Pass 3: Generate trimmed masks"""
    if verbose:
        print("STAGE: Pass 3 - Trimmed masks")
    
    iterator = tqdm(example_dirs) if verbose else example_dirs
    for example_dir in iterator:
        try:
            example_dir.trimmed_mask_path()  # This generates trimmed mask if missing
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate trimmed mask for {example_dir.d}: {e}")


def generate_negmask_probs(example_dirs: List[ExampleDir], verbose: bool):
    """Pass 4: Generate negmask probabilities"""
    if verbose:
        print("STAGE: Pass 4 - Negmask probabilities")
    
    iterator = tqdm(example_dirs) if verbose else example_dirs
    for example_dir in iterator:
        try:
            example_dir.trash_probs_path()  # This generates both trash and other probs
            example_dir.other_probs_path()
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to generate negmask probabilities for {example_dir.d}: {e}")

