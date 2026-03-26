from typing import Callable
from pathlib import Path
from tqdm import tqdm
from .core import ExampleDir


def generate_smallnet_50x50(directories, verbose=True):
    example_dirs = to_example_dirs(directories, verbose)

    def _itgen(generator, title):
        iterate_and_generate(
            example_dirs,
            verbose,
            generator,
            title,
        )
    _itgen(lambda edir: edir.smallnet_50x50_path(), "Pass 1 - 50x50 Smallnet masks")
    

def run_bulk_inference(directories: list[Path | str | ExampleDir], smallnet, negmask, verbose=True):
    """
    Run bulk inference on a list of directories in passes.

    Args:
        directories: list of directory paths that contain image.jpg files
        smallnet_learner: Optional smallnet learner, defaults to get_default_smallnet_learner()
        negmask_learner: Optional negmask learner, defaults to get_default_negmask_learner()
        flower_learner: Optional flower learner, defaults to get_default_flower_learner()
        mapillary_segformer: Optional mapillary segformer
        elev_segformer: Optional elevation segformer
        verbose: Whether to show progress bars and stage information

    Returns:
        list[ExampleDir]: list of ExampleDir instances for further processing
    """
    example_dirs = to_example_dirs(directories, smallnet, negmask, verbose)

    def _itgen(generator, title):
        iterate_and_generate(
            example_dirs,
            verbose,
            generator,
            title,
        )

    _itgen(lambda edir: edir.smallnet_mask_path(), "Pass 1 - 100x100 Smallnet masks")
    _itgen(lambda edir: edir.mapi_mask_path(), "Pass 2 - Mapillary Mask paths")
    _itgen(lambda edir: edir.elev_mask_path(), "Pass 3 - Elev Mask paths")
    _itgen(lambda edir: edir.trimmed_mask_path(), "Pass 4 - 100x100 trimmed mask paths")
    _itgen(lambda edir: edir.negmask_100x100_paths(), "Pass 5 - 100x100 negmasks")
    _itgen(lambda edir: edir.flower_pos_probs_path(), "Pass 6 - flower pos")
    _itgen(lambda edir: edir.flower_neg_probs_path(), "Pass 7 - flower neg")

    if verbose:
        print(f"Bulk inference complete for {len(example_dirs)} directories")

    return example_dirs


def to_example_dirs(directories, smallnet, negmask, verbose):
    if verbose:
        print("Setting up ExampleDir instances...")
    example_dirs = []
    for directory in directories:
        try:
            if isinstance(directory, ExampleDir):
                example_dir = directory
            else:
                example_dir = ExampleDir(directory, smallnet, negmask)
            example_dirs.append(example_dir)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not create ExampleDir for {directory}: {e}")
    return example_dirs


def iterate_and_generate(
    example_dirs: list[ExampleDir],
    verbose: bool,
    generator: Callable[[ExampleDir], None],
    title: str,
):
    if verbose:
        print(f"STAGE: {title}")

    iterator = tqdm(example_dirs) if verbose else example_dirs
    for example_dir in iterator:
        try:
            generator(example_dir)
        except Exception as e:
            if verbose:
                print(f"Warning: failure in stage={title}\n{example_dir.d}: {e}")


# def generate_smallnet_masks(example_dirs: list[ExampleDir], verbose: bool):
#     iterate_and_generate(
#         example_dirs,
#         verbose,
#         lambda edir: edir.smallnet_mask_path(),
#         "Pass 1 - Smallnet masks",
#     )
#     # """Pass 1: Generate smallnet masks for all images"""
#     # if verbose:
#     #     print("STAGE: Pass 1 - Smallnet masks")

#     # iterator = tqdm(example_dirs) if verbose else example_dirs
#     # for example_dir in iterator:
#     #     try:
#     #         example_dir.smallnet_mask_path()  # This generates the mask if missing
#     #     except Exception as e:
#     #         if verbose:
#     #             print(
#     #                 f"Warning: Failed to generate smallnet mask for {example_dir.d}: {e}"
#     #             )


# def generate_segmentation_masks(example_dirs: list[ExampleDir], verbose: bool):
#     """Pass 2: Generate segmentation masks (mapillary and elevation)"""
#     if verbose:
#         print("STAGE: Pass 2 - Segmentation masks")

#     iterator = tqdm(example_dirs) if verbose else example_dirs
#     for example_dir in iterator:
#         try:
#             example_dir.mapi_mask_path()  # This generates mapi mask if missing
#             example_dir.elev_mask_path()  # This generates elev mask if missing
#         except Exception as e:
#             if verbose:
#                 print(
#                     f"Warning: Failed to generate segmentation masks for {example_dir.d}: {e}"
#                 )


# def generate_trimmed_masks(example_dirs: list[ExampleDir], verbose: bool):
#     """Pass 3: Generate trimmed masks"""
#     if verbose:
#         print("STAGE: Pass 3 - Trimmed masks")

#     iterator = tqdm(example_dirs) if verbose else example_dirs
#     for example_dir in iterator:
#         try:
#             example_dir.trimmed_mask_path()  # This generates trimmed mask if missing
#         except Exception as e:
#             if verbose:
#                 print(
#                     f"Warning: Failed to generate trimmed mask for {example_dir.d}: {e}"
#                 )


# def generate_negmask_probs(example_dirs: list[ExampleDir], verbose: bool):
#     """Pass 4: Generate negmask probabilities"""
#     if verbose:
#         print("STAGE: Pass 4 - Negmask probabilities")

#     iterator = tqdm(example_dirs) if verbose else example_dirs
#     for example_dir in iterator:
#         try:
#             example_dir.trash_probs_path()  # This generates both trash and other probs
#             example_dir.other_probs_path()
#         except Exception as e:
#             if verbose:
#                 print(
#                     f"Warning: Failed to generate negmask probabilities for {example_dir.d}: {e}"
#                 )


# def generate_flower_probs(example_dirs: list[ExampleDir], verbose: bool):
#     """Pass 5: Generate flower probabilities"""
#     if verbose:
#         print("STAGE: Pass 5 - Flower probabilities")

#     iterator = tqdm(example_dirs) if verbose else example_dirs
#     for example_dir in iterator:
#         try:
#             example_dir.flower_pos_probs_path()  # This generates both pos and neg probs
#             example_dir.flower_neg_probs_path()
#         except Exception as e:
#             if verbose:
#                 print(
#                     f"Warning: Failed to generate flower probabilities for {example_dir.d}: {e}"
#                 )
