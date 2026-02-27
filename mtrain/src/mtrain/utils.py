from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import string
import json


def mkdir(p: Path):
    p.mkdir(exist_ok=True, parents=True)
    return p


def json_to_content(json_path_or_content) -> dict:
    content = json_path_or_content
    if isinstance(json_path_or_content, Path) or isinstance(json_path_or_content, str):
        with open(json_path_or_content) as f:
            content = json.load(f)
    return content


def compose(*funcs):
    def composed(x):
        for f in reversed(funcs):
            x = f(x)
        return x
    return composed


def pipe(*funcs):
    def piped(x):
        for f in funcs:
            x = f(x)
        return x
    return piped

def random_filename(k, suffix=None):
    # pathlib style suffix (inclues ., like ".jpg")
    chars = string.ascii_lowercase + string.digits
    name = "".join(random.choices(chars, k=k))
    if suffix is None:
        return name
    else:
        return f"{name}{suffix}"

def show(crops, figsize=None, ncols=2):
    crops = list(crops)
    rows = math.ceil(len(crops) / ncols)
    if figsize is None:
        figsize = (10 * rows, 10 * rows)
        print("figsize", figsize)
    _, axs = plt.subplots(rows, ncols, figsize=figsize)
    if len(crops) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    for i, c in enumerate(crops):
        axs[i].imshow(c)
    plt.tight_layout()
    plt.show()


def draw_grid(ax, img_shape, cell_size, color="white", lw=0.5, alpha=0.5):
    h, w = img_shape[:2]
    ax.set_xticks(np.arange(0, w, cell_size), minor=True)
    ax.set_yticks(np.arange(0, h, cell_size), minor=True)

    # vertical lines
    for x in range(0, w, cell_size):
        ax.axvline(x, color=color, lw=lw, alpha=alpha)

    # horizontal lines
    for y in range(0, h, cell_size):
        ax.axhline(y, color=color, lw=lw, alpha=alpha)