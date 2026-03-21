import pytest
import shutil
from pathlib import Path
import numpy as np
from mtrain.example_dir import ExampleDir
from mtrain.example_dir.learners import SmallnetLearner, NegmaskLearner, get_raw_smallnet_learner, get_raw_negmask_learner, step_downer
from mtrain.example_dir.core import load_npz
from mtrain.disk import DiskBooleanMask

# Template and model paths
TEMPLATE_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/unit-tests/example-dir/template")
SMALLNET_MODEL_PATH = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/unit-tests/smallnet/test-model-tile_128x128-xresnet18.pth")
NEGMASK_MODEL_PATH = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/unit-tests/negmask/test-model_cropsize-224_xresnet18_tfm-stepedge.pth")


@pytest.fixture(scope="module")
def test_smallnet_learner():
    """Create test SmallnetLearner instance"""
    # Create dummy data dir for raw learner initialization
    TEST_DATA_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/models/dummy_smallnet_data")
    
    learner = get_raw_smallnet_learner(
        tile_size=128,
        bs=4,
        data_dir=TEST_DATA_DIR,
        pth_path=SMALLNET_MODEL_PATH,
        arch="xresnet18"
    )
    
    return SmallnetLearner(
        label="test",
        learner=learner,
        bs=4,
        tile_size=128,
        strides=[64],
        area_low=20,
        area_high=None
    )


@pytest.fixture(scope="module") 
def test_negmask_learner():
    """Create test NegmaskLearner instance"""
    learner = get_raw_negmask_learner(
        bs=4,
        crop_size=224,
        pth_path=NEGMASK_MODEL_PATH,
        arch="xresnet18"
    )
    
    return NegmaskLearner(
        label="test",
        learner=learner,
        bs=4,
        crop_size=224,
        bbox_pad=10,
        mutator=step_downer
    )


def copy_template_to_tmp(tmp_path: Path, suffix: str = "") -> list[Path]:
    """Copy template directories to tmp directory for testing
    
    Args:
        tmp_path: pytest tmp_path fixture
        suffix: optional suffix to add to work directory name
        
    Returns:
        List of copied directory paths
    """
    work_dir = tmp_path / f"work{suffix}"
    work_dir.mkdir()
    
    copied_dirs = []
    subdirs = sorted(TEMPLATE_DIR.iterdir())
    for template_subdir in subdirs:
        if template_subdir.is_dir() and (template_subdir / "image.jpg").exists():
            dest_dir = work_dir / template_subdir.name
            shutil.copytree(template_subdir, dest_dir)
            copied_dirs.append(dest_dir)
    
    return copied_dirs


# def test_example_dir_creation(tmp_path, test_smallnet_learner, test_negmask_learner):
#     """Test basic ExampleDir creation and initialization"""
#     # Copy template to tmp directory
#     test_dirs = copy_template_to_tmp(tmp_path)
#     test_dir = test_dirs[0]
    
#     # Create learner dictionaries
#     smallnet_learners = {"test": test_smallnet_learner}
#     negmask_learners = {"test": test_negmask_learner}
    
#     # Create ExampleDir
#     edir = ExampleDir(test_dir, smallnet_learners, negmask_learners)
    
#     # Basic checks
#     assert edir.image_path.exists()
#     assert edir.image_path.name == "image.jpg"
#     assert edir.d == test_dir
#     assert "test" in edir.label_by_smallnet
#     assert "test" in edir.label_by_negmask


def test_smallnet_mask_generation(tmp_path, test_smallnet_learner, test_negmask_learner):
    """Test smallnet mask generation for single ExampleDir"""
    # Copy template to tmp directory
    test_dirs = copy_template_to_tmp(tmp_path, "_single")
    test_dir = test_dirs[0]
    
    # Create learner dictionaries
    smallnet_learners = {"test": test_smallnet_learner}
    negmask_learners = {"test": test_negmask_learner}
    
    # Create ExampleDir
    edir = ExampleDir(test_dir, smallnet_learners, negmask_learners)
    
    # Generate smallnet mask
    mask_path = edir.smallnet_mask_path("test")
    
    # Check file was created
    assert mask_path.exists()
    assert mask_path.suffix == ".png"
    assert mask_path.name == "mask-test.png"
    
    # Load and check basic properties
    mask = DiskBooleanMask.load(mask_path)
    assert isinstance(mask, np.ndarray)
    # Image should be resized to max 1024x1024 if larger
    assert mask.shape[0] <= 1024
    assert mask.shape[1] <= 1024


def test_batch_vs_individual_smallnet_prediction(tmp_path, test_smallnet_learner, test_negmask_learner):
    """Test that batch prediction produces same results as individual predictions for both smallnet and negmask"""
    # Create separate work directories for batch and individual tests
    batch_dirs = copy_template_to_tmp(tmp_path, "_batch")[:4]  # Use first 4 dirs
    individual_dirs = copy_template_to_tmp(tmp_path, "_individual")[:4]
    
    # Create learner dictionaries
    smallnet_learners = {"test": test_smallnet_learner}
    negmask_learners = {"test": test_negmask_learner}
    
    # Create ExampleDir instances
    batch_edirs = [ExampleDir(d, smallnet_learners, negmask_learners) for d in batch_dirs]
    individual_edirs = [ExampleDir(d, smallnet_learners, negmask_learners) for d in individual_dirs]
    
    # Test smallnet batch vs individual prediction
    # Run batch prediction
    ExampleDir.batch_predict_smallnet_masks(test_smallnet_learner, batch_edirs)
    
    # Run individual predictions
    for edir in individual_edirs:
        edir.smallnet_mask_path("test")
    
    # Compare smallnet results
    for batch_edir, individual_edir in zip(batch_edirs, individual_edirs):
        # Load both masks
        batch_mask_path = batch_edir._get_smallnet_mask_path("test")
        individual_mask_path = individual_edir._get_smallnet_mask_path("test")
        
        assert batch_mask_path.exists()
        assert individual_mask_path.exists()
        
        batch_mask = DiskBooleanMask.load(batch_mask_path)
        individual_mask = DiskBooleanMask.load(individual_mask_path)
        
        # Masks should be identical
        assert batch_mask.shape == individual_mask.shape
        assert np.array_equal(batch_mask, individual_mask), f"Smallnet masks differ for {batch_edir.d.name}"
    
    # Test negmask batch vs individual prediction (reusing smallnet results)
    
    # Run negmask batch prediction
    ExampleDir.batch_predict_negmask_masks(test_negmask_learner, batch_edirs, "test", 4)
    
    # Run individual negmask predictions
    for edir in individual_edirs:
        edir.negmask_paths("test", "test")
    
    # Compare negmask results
    for batch_edir, individual_edir in zip(batch_edirs, individual_edirs):
        # Get paths for both other and trash probabilities
        batch_other_path, batch_trash_path = batch_edir.negmask_paths("test", "test")
        individual_other_path, individual_trash_path = individual_edir.negmask_paths("test", "test")
        
        # Check all files exist
        assert batch_other_path.exists()
        assert batch_trash_path.exists()
        assert individual_other_path.exists()
        assert individual_trash_path.exists()
        
        # Load and compare other probabilities
        batch_other = load_npz(batch_other_path)
        individual_other = load_npz(individual_other_path)
        assert batch_other.shape == individual_other.shape
        assert np.allclose(batch_other, individual_other, rtol=1e-5), f"Other probs differ for {batch_edir.d.name}"
        
        # Load and compare trash probabilities
        batch_trash = load_npz(batch_trash_path)
        individual_trash = load_npz(individual_trash_path)
        assert batch_trash.shape == individual_trash.shape
        assert np.allclose(batch_trash, individual_trash, rtol=1e-5), f"Trash probs differ for {batch_edir.d.name}"


def test_invalid_directory():
    """Test ExampleDir creation with invalid directory"""
    with pytest.raises(Exception, match="image.*does not exist"):
        ExampleDir("/nonexistent/path", {}, {})