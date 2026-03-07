from pathlib import Path
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import string
import json
import cv2


def mkdir(p: Path | str):
    Path(p).mkdir(exist_ok=True, parents=True)
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

def show(crops, figsize=None, ncols=2, axis="on"):
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
        axs[i].axis(axis)
    plt.tight_layout()
    plt.show()


def draw_grid_cv2(img, cell_size, color=(255, 255, 255), thickness=1, alpha=0.5):
    """Draw a grid on the image array using cv2.
    
    Args:
        img: numpy array (H, W, 3) or (H, W)
        cell_size: size of grid cells in pixels
        color: tuple of BGR values, default white (255, 255, 255)
        thickness: line thickness in pixels
        alpha: blending factor for transparency (0-1)
    
    Returns:
        Image array with grid drawn on it
    """
    result = img.copy().astype(np.float32)
    h, w = img.shape[:2]
    
    # Create a transparent grid overlay
    grid_overlay = np.zeros_like(result)
    
    # Draw vertical lines
    for x in range(0, w, cell_size):
        cv2.line(grid_overlay, (x, 0), (x, h), color, thickness)
    
    # Draw horizontal lines
    for y in range(0, h, cell_size):
        cv2.line(grid_overlay, (0, y), (w, y), color, thickness)
    
    # Blend the grid overlay with the original image
    result = (1 - alpha) * result + alpha * grid_overlay
    
    return result.astype(np.uint8)


def it_chain(iterator):
    return list(itertools.chain.from_iterable(iterator))


def overlay_mask_on_img(img_arr, mask, alpha=0.4, color=[255,0,0]):
    res = img_arr.copy()
    mask = mask.astype(bool)
    res[mask] = (
        (1 - alpha) * res[mask].astype(np.float32) +
        alpha * np.array(color)
    ).astype(np.uint8)
    return res

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
