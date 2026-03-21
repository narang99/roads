from mtrain.utils import globL
from mtrain.disk import DiskImage, DiskBooleanMask
import pytest
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple
import itertools

from mtrain.neg_mask.model.predict.trash import (
    batch_predict_and_return_prob_masks,
    predict_and_return_prob_masks_using_unblurred,
    get_learner,
    id_mutator,
    get_crops_masks_bboxes,
)
from mtrain.neg_mask.model.datasets.blur_pad_dl import (
    BlurPadInferDataset,
    blur_overwriter,
)


TEST_DATA_DIR = Path(
    "/Users/hariomnarang/Desktop/personal/roads/datasets/unit-tests/negmask/smallnet-results"
)
IMAGES_DIR = TEST_DATA_DIR / "train"
MASKS_DIR = TEST_DATA_DIR / "masks"
CROP_SIZE = 224
BBOX_PAD = 10
BS = 4


def _load(image_path, masks_dir):
    mask_file = MASKS_DIR / f"{image_path.stem}.png"
    image = DiskImage.load(image_path)
    mask = DiskBooleanMask.load(mask_file)
    return image, mask


def get_test_image_mask_pairs(n_images: int = 5, add_empty_masks: int = 2):
    images, masks = [], []
    image_files = sorted(globL(IMAGES_DIR, "*.jpg"))[:n_images]

    for image_path in image_files:
        image, mask = _load(image_path, MASKS_DIR)
        images.append(image)
        masks.append(mask)

    for i in range(add_empty_masks):
        image = DiskImage.load(image_files[0])
        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        images.append(image)
        masks.append(empty_mask)

    return images, masks


@pytest.fixture(scope="module")
def learner():
    """Create a learner instance for testing."""
    learn = get_learner()
    # Initialize model weights for reproducible testing
    torch.manual_seed(42)
    for param in learn.model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    return learn


# @pytest.fixture(scope="module")
def get_test_data():
    """Load test images and masks."""
    return get_test_image_mask_pairs(n_images=3, add_empty_masks=1)


def masks_are_close(
    mask1: np.ndarray, mask2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-7
) -> bool:
    """Check if two masks are close enough."""
    return np.allclose(mask1, mask2, rtol=rtol, atol=atol)


def test_dataset_output(learner):
    """Test what the dataset actually returns."""
    # Create simple test data
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:150, 50:150] = 255  # Add a square region

    crops, masks_out, bboxes, inner_bboxes = get_crops_masks_bboxes(
        image, mask, crop_size=130, bbox_pad=5
    )

    assert len(bboxes) == 1

    ds = BlurPadInferDataset(
        crops,
        masks_out,
        inner_bboxes,
        130,
        id_mutator,
    )

    # Test individual item
    item = ds[0]

    # Test DataLoader
    dl = DataLoader(ds, batch_size=2)
    batch = next(iter(dl))

    # This test just verifies shapes are correct
    assert len(item.shape) == 3, f"Expected 3D tensor, got {item.shape}"
    assert item.shape[0] == 3, f"Expected 3 channels, got {item.shape[0]}"
    assert len(batch.shape) == 4, f"Expected 4D batch tensor, got {batch.shape}"

def test_batch_predictions_basic_functionality(learner):
    """Test that batch predictions work and return correct shapes."""
    images, masks = get_test_data()

    single_results = []
    for image, mask in zip(images, masks):
        res = predict_and_return_prob_masks_using_unblurred(
            image,
            mask,
            learner,
            crop_size=CROP_SIZE,
            bbox_pad=BBOX_PAD,
            mutator=blur_overwriter(1, 1),
        )
        single_results.append(res)

    # Get batch predictions
    batch_results = batch_predict_and_return_prob_masks(
        images,
        masks,
        learner,
        crop_size=CROP_SIZE,
        bbox_pad=BBOX_PAD,
        mutator=blur_overwriter(1, 1),
        bs=BS,
    )

    assert len(single_results) == len(batch_results)
    assert len(batch_results) == len(images)
    for s, b in zip(single_results, batch_results):
        os, ts = s
        ob, tb = s
        assert np.allclose(os, ob)
        assert np.allclose(ts, tb)


def test_empty_masks_handling(learner):
    """Test that empty masks are handled correctly."""
    # Create test data with only empty masks
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    empty_mask = np.zeros((256, 256), dtype=np.uint8)

    # Test batch prediction with empty mask
    batch_results = batch_predict_and_return_prob_masks(
        [image],
        [empty_mask],
        learner,
        crop_size=130,
        bbox_pad=5,
        mutator=id_mutator,
    )

    single_results = predict_and_return_prob_masks_using_unblurred(
        image,
        empty_mask,
        learner,
        crop_size=130,
        bbox_pad=5,
        mutator=id_mutator,
    )


    assert len(batch_results) == 1
    batch_other, batch_trash = batch_results[0]

    # Should return zero masks for empty input
    assert np.all(batch_other == 0), "Batch other mask should be all zeros for empty mask"
    assert np.all(batch_trash == 0), "Batch trash mask should be all zeros for empty mask"

    batch_other, batch_trash = single_results

    # Should return zero masks for empty input
    assert np.all(batch_other == 0), "Single other mask should be all zeros for empty mask"
    assert np.all(batch_trash == 0), "Single trash mask should be all zeros for empty mask"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
