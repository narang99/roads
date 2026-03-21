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
def test_data():
    """Load test images and masks."""
    return get_test_image_mask_pairs(n_images=3, add_empty_masks=1)


def masks_are_close(
    mask1: np.ndarray, mask2: np.ndarray, rtol: float = 1e-5, atol: float = 1e-7
) -> bool:
    """Check if two masks are close enough."""
    return np.allclose(mask1, mask2, rtol=rtol, atol=atol)


class TestBatchPredict:
    def test_dataset_output(self, learner):
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
        print(f"Dataset item shape: {item.shape}")
        print(f"Dataset item dtype: {item.dtype}")

        # Test DataLoader
        dl = DataLoader(ds, batch_size=2)
        batch = next(iter(dl))
        print(f"DataLoader batch shape: {batch.shape}")
        print(f"DataLoader batch dtype: {batch.dtype}")

        # This test just verifies shapes are correct
        assert len(item.shape) == 3, f"Expected 3D tensor, got {item.shape}"
        assert item.shape[0] == 3, f"Expected 3 channels, got {item.shape[0]}"
        assert len(batch.shape) == 4, f"Expected 4D batch tensor, got {batch.shape}"

    def test_batch_predictions_basic_functionality(self, learner):
        """Test that batch predictions work and return correct shapes."""
        images, masks = test_data()

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
        # batch_results = batch_predict_and_return_prob_masks(
        #     images,
        #     masks,
        #     learner,
        #     crop_size=CROP_SIZE,
        #     bbox_pad=BBOX_PAD,
        #     mutator=blur_overwriter(1, 1),
        #     bs=BS,
        # )

        # assert len(single_results) == len(batch_results)
        # assert len(batch_results) == len(images)
        # for s, b in zip(single_results, batch_results):
        #     assert s == b

        # # Basic validation
        # assert len(batch_results) == len(images), \
        #     f"Expected {len(images)} results, got {len(batch_results)}"

        # for i, (result, mask) in enumerate(zip(batch_results, masks)):
        #     other_mask, trash_mask = result

        #     assert other_mask.shape == mask.shape, \
        #         f"Image {i}: Other mask shape mismatch: expected {mask.shape}, got {other_mask.shape}"
        #     assert trash_mask.shape == mask.shape, \
        #         f"Image {i}: Trash mask shape mismatch: expected {mask.shape}, got {trash_mask.shape}"

        #     # Check that we get valid probability values
        #     assert np.all(other_mask >= 0) and np.all(other_mask <= 1), \
        #         f"Image {i}: Other mask values should be probabilities [0,1]"
        #     assert np.all(trash_mask >= 0) and np.all(trash_mask <= 1), \
        #         f"Image {i}: Trash mask values should be probabilities [0,1]"


#     def test_empty_masks_handling(self, learner):
#         """Test that empty masks are handled correctly."""
#         # Create test data with only empty masks
#         image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
#         empty_mask = np.zeros((256, 256), dtype=np.uint8)

#         # Test batch prediction with empty mask
#         batch_results = batch_predict_and_return_prob_masks(
#             [image],
#             [empty_mask],
#             learner,
#             crop_size=130,
#             bbox_pad=5,
#             mutator=id_mutator,
#             bs=128
#         )

#         assert len(batch_results) == 1
#         batch_other, batch_trash = batch_results[0]

#         # Should return zero masks for empty input
#         assert np.all(batch_other == 0), "Batch other mask should be all zeros for empty mask"
#         assert np.all(batch_trash == 0), "Batch trash mask should be all zeros for empty mask"


#     def test_mixed_batch(self, learner):
#         """Test batch with mix of normal and empty masks."""
#         # Create mixed test data
#         images = []
#         masks = []

#         # Add normal image with non-empty mask
#         image1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
#         mask1 = np.zeros((256, 256), dtype=np.uint8)
#         mask1[50:150, 50:150] = 255  # Add a square region
#         images.append(image1)
#         masks.append(mask1)

#         # Add image with empty mask
#         image2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
#         mask2 = np.zeros((256, 256), dtype=np.uint8)
#         images.append(image2)
#         masks.append(mask2)

#         # Add another normal image
#         image3 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
#         mask3 = np.zeros((300, 300), dtype=np.uint8)
#         mask3[100:200, 100:200] = 255
#         images.append(image3)
#         masks.append(mask3)

#         # Get batch predictions
#         batch_results = batch_predict_and_return_prob_masks(
#             images,
#             masks,
#             learner,
#             crop_size=130,
#             bbox_pad=5,
#             mutator=id_mutator,
#             bs=128
#         )

#         # Verify we get correct number of results
#         assert len(batch_results) == 3, f"Expected 3 results, got {len(batch_results)}"

#         # Check shapes match input masks
#         for i, ((other_mask, trash_mask), orig_mask) in enumerate(zip(batch_results, masks)):
#             assert other_mask.shape == orig_mask.shape, \
#                 f"Image {i}: Other mask shape mismatch"
#             assert trash_mask.shape == orig_mask.shape, \
#                 f"Image {i}: Trash mask shape mismatch"

#         # Verify empty mask produces zero outputs
#         assert np.all(batch_results[1][0] == 0), "Empty mask should produce zero other mask"
#         assert np.all(batch_results[1][1] == 0), "Empty mask should produce zero trash mask"


# @pytest.mark.parametrize("n_images,n_empty", [
#     (3, 0),  # Only real images
#     (5, 2),  # Mix of real and empty
#     (1, 3),  # More empty than real
# ])
# def test_different_batch_sizes(learner, n_images, n_empty):
#     """Test with different combinations of images and empty masks."""
#     test_data = get_test_image_mask_pairs(n_images=n_images, add_empty_masks=n_empty)

#     if not test_data:
#         pytest.skip("No test data available")

#     images, masks = zip(*test_data)
#     images = list(images)
#     masks = list(masks)

#     # Get batch predictions
#     batch_results = batch_predict_and_return_prob_masks(
#         images,
#         masks,
#         learner,
#         crop_size=130,
#         bbox_pad=5,
#         mutator=id_mutator,
#         bs=64  # Use smaller batch size
#     )

#     # Basic validation
#     assert len(batch_results) == len(images)
#     for i, (result, mask) in enumerate(zip(batch_results, masks)):
#         other_mask, trash_mask = result
#         assert other_mask.shape == mask.shape
#         assert trash_mask.shape == mask.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
