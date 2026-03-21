import pytest
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List
from mtrain.smallnet.unet.predict.strided.single import strided_predict_unet_only_mask as single_strided
from mtrain.smallnet.unet.predict.strided.multiple import strided_predict_unet_only_mask as multiple_strided
from mtrain.example_dir.learners.smallnet import get_smallnet_learner
from mtrain.disk import DiskImage


TEST_IMAGES_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/unit-tests/smallnet/images")
TEST_DATA_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/models/dummy_smallnet_data")
TEST_MODEL_PATH = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/unit-tests/smallnet/test-model-tile_128x128-xresnet18.pth")


def load_test_images(n_images: int = 4) -> list[np.ndarray]:
    """Load test images from the specified directory."""
    image_files = sorted(list(TEST_IMAGES_DIR.glob("*.jpg")))[:n_images]
    
    if len(image_files) < n_images:
        raise ValueError(f"Only {len(image_files)} images found, need {n_images}")
    
    images = []
    for img_path in image_files:
        disk_img = DiskImage.load(img_path)
        resized_img = cv2.resize(disk_img, (512,512), interpolation=cv2.INTER_AREA)
        images.append(resized_img)
    
    return images


@pytest.fixture(scope="module")
def smallnet_learner():
    """Create a real smallnet learner for testing."""
    tile_size = 128
    bs = 4
    return get_smallnet_learner(
        tile_size=tile_size,
        bs=bs, 
        data_dir=TEST_DATA_DIR,
        pth_path=TEST_MODEL_PATH,
        arch="xresnet18"
    )


def test_strided_single_vs_batched_output(smallnet_learner):
    """Test that single strided and batched strided produce the same output."""
    test_images = load_test_images(4)
    tile_size = 128
    strides = [32, 64]
    bs = 4
    
    # Get single predictions
    single_results = []
    for img_arr in test_images:
        result = single_strided(
            img_arr=img_arr,
            tile_size=tile_size,
            learner=smallnet_learner,
            strides=strides,
            bs=bs
        )
        single_results.append(result)
    
    # Get batched prediction
    batched_results = multiple_strided(
        img_arrs=test_images,
        tile_size=tile_size,
        learner=smallnet_learner,
        strides=strides,
        bs=bs
    )
    
    # Compare results
    assert len(single_results) == len(batched_results), "Number of results should match"
    assert len(batched_results) == len(test_images), "Should have one result per input image"
    
    for i, (single_mask, batched_mask) in enumerate(zip(single_results, batched_results)):
        assert single_mask.shape == batched_mask.shape, f"Shape mismatch for image {i}"
        assert single_mask.dtype == batched_mask.dtype, f"Dtype mismatch for image {i}"
        
        # Check if masks are exactly identical - they should be with same model and deterministic operations
        assert np.array_equal(single_mask, batched_mask), f"Masks should be identical for image {i}"
        
        # Check output shape matches input
        assert single_mask.shape == test_images[i].shape[:2], f"Output shape should match input for image {i}"



def test_strided_parameters(smallnet_learner):
    """Test different stride configurations."""
    test_images = load_test_images()
    tile_size = 128
    test_configurations = [
        {"strides": [], "bs": 2},
        {"strides": [32], "bs": 2},
        {"strides": [16, 32, 48], "bs": 4},
    ]
    
    # Take just first image for parameter testing
    test_img = test_images[0]
    
    for config in test_configurations:
        # Test single version
        single_result = single_strided(
            img_arr=test_img,
            tile_size=tile_size,
            learner=smallnet_learner,
            **config
        )
        
        # Test batched version
        batched_results = multiple_strided(
            img_arrs=[test_img],
            tile_size=tile_size,
            learner=smallnet_learner,
            **config
        )
        
        assert len(batched_results) == 1, "Should get one result for one image"
        batched_result = batched_results[0]
        
        assert single_result.shape == batched_result.shape, f"Shape mismatch for config {config}"
        assert single_result.dtype == batched_result.dtype, f"Dtype mismatch for config {config}"
        assert np.array_equal(single_result, batched_result), f"Results should be identical for config {config}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])