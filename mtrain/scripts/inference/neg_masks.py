import argparse
from pathlib import Path

from mtrain.neg_mask.model.predict.full_image_8chan import (
    predict_and_return_probs,
)
from fastai.vision.all import (
    vision_learner,
    DataLoaders,
    resnet18,
    accuracy,
    F1Score,
    CrossEntropyLossFlat,
    ProgressCallback,
)
from tqdm import tqdm
import numpy as np
from mtrain.disk import DiskImage, DiskBooleanMask
from mtrain.neg_mask.model.learner import dummy_dls
from mtrain.neg_mask.model.crop_level_dataset import CropLevelDataset2Chan

NEG_MASK_MODEL_PATH = Path(
    "/Users/hariomnarang/Desktop/personal/roads/datasets/models/trash_classification/resnet18-size_220-chan_8-with_augs-iter_15"
)


def _load_neg_mask_model(model_path):
    LABELS = ["other", "trash"]
    DataLoaders.from_dsets(
        CropLevelDataset2Chan([], LABELS, True, medium_pad=220),
        CropLevelDataset2Chan([], LABELS, False, medium_pad=220),
    )

    learner = vision_learner(
        dummy_dls(LABELS),
        resnet18,
        n_in=8,
        metrics=[accuracy, F1Score(average="macro")],
        loss_func=CrossEntropyLossFlat(),
        n_out=len(LABELS),
        normalize=False,
    )
    learner = learner.remove_cb(ProgressCallback)

    learner = learner.load(NEG_MASK_MODEL_PATH)
    return learner


def _save_neg_mask_probs(pred_dirs, mask_name: str = "mask.png"):
    print(f"STAGE: negmask, total directories={len(pred_dirs)}")
    learner = _load_neg_mask_model(NEG_MASK_MODEL_PATH)
    for d in tqdm(pred_dirs):
        image, mask = (
            DiskImage.load(d / "image.jpg"),
            DiskBooleanMask.load(d / mask_name),
        )
        trash_probs, other_probs = predict_and_return_probs(
            image,
            mask,
            learner,
            220,
        )
        np.save(d / "trash_probs.npy", trash_probs)
        np.save(d / "other_probs.npy", other_probs)


def save_neg_masks_probs(root_dir: Path, mask_name: str = "mask.png"):
    total_dirs = [d for d in root_dir.glob("*")]
    dirs = [d for d in total_dirs if (d / "image.jpg").exists() and (d / mask_name).exists()]
    print("STAGE: negative masks: total directories =", len(total_dirs), "filtered =", len(dirs))
    _save_neg_mask_probs(dirs, mask_name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate negative mask probabilities for images in place"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing subdirectories with image.jpg and mask files",
    )
    parser.add_argument(
        "--mask-name",
        type=str,
        default="mask.png",
        help="Name of the mask file within each subdirectory (default: mask.png)",
    )
    
    args = parser.parse_args()
    save_neg_masks_probs(args.root_dir, args.mask_name)


if __name__ == "__main__":
    main()
