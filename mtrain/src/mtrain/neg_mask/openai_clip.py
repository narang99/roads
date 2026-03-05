from pathlib import Path
import itertools
from typing import Iterator


def get_images_from_clip_file(path) -> Iterator[Path]:
    with open(path) as f:
        lines = f.readlines()
    return (Path(line.split("\t")[1].strip()) for line in lines)

def get_image_dirs_from_clip_file(path) -> list[tuple[str, Path]]:
    return [(path.stem, p.parent) for p in get_images_from_clip_file(path)]


def interleaved_data_from_multiple_clip_files(paths) -> Iterator[tuple[str, Path]]:
    all_cat_dirs = [get_image_dirs_from_clip_file(p) for p in paths]
    return itertools.chain.from_iterable(itertools.zip_longest(*all_cat_dirs))
