from .core import ExampleDir, get_default_smallnet_learner, get_default_negmask_learner
from pathlib import Path
import shutil
from .bulk import run_bulk_inference

__all__ = ["ExampleDir", "get_default_smallnet_learner", "get_default_negmask_learner", "run_bulk_inference"]


def create_dirs_for_images(images: list[Path | str], dest_dir: Path):
    """Setup directory structure and copy images"""
    directories = []
    for img in images:
        img = Path(img)
        dest = dest_dir / img.stem
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / "image.jpg")
        directories.append(dest)
    return directories