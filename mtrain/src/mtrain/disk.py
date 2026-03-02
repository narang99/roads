from PIL import Image
import numpy as np
from pathlib import Path


class DiskImage:
    @classmethod
    def save(cls, arr: np.ndarray, path: Path | str):
        Image.fromarray(arr, "RGB").save(path)

    @classmethod
    def load(cls, path: Path | str):
        return np.array(Image.open(path).convert("RGB"))


class DiskBooleanMask:
    @classmethod
    def save(cls, arr: np.ndarray, path: Path | str):
        Image.fromarray(arr, "L").save(path)

    @classmethod
    def load(cls, path: Path | str):
        return np.array(Image.open(path).convert("L"))

    @classmethod
    def load_as_bool(cls, path: Path | str):
        return np.array(Image.open(path).convert("L")).astype(bool)
