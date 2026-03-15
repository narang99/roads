from pathlib import Path
from typing import Callable


def has_image_jpg(d: Path):
    return (d / "image.jpg").exists()


def has_mask_png(d: Path):
    return (d / "mask.png").exists()


def has_meta_json(d: Path):
    return (d / "meta.json").exists()


def has_source_dir(d: Path):
    return (d / "source_dir").exists() and (d / "source_dir" / "image.jpg").exists()


def and_fns(filter_funcs: list[Callable[[Path], bool]]):
    def wrapped(d):
        res = [fn(d) for fn in filter_funcs]
        return all(res)

    return wrapped


def or_fns(filter_funcs: list[Callable[[Path], bool]]):
    def wrapped(d):
        res = [fn(d) for fn in filter_funcs]
        return any(res)

    return wrapped


def is_dir(d: Path):
    return d.is_dir()


def get_dirs(root_dir, filter_fn=is_dir):
    return filter(filter_fn, root_dir.glob("*"))


def get_labelled_dirs(root_dir, filter_fn=is_dir, labels=None):
    if labels is None:
        labels = ["other", "trash"]
    for label in labels:
        d = root_dir / label
        for d in get_dirs(d, filter_fn):
            yield d, label
