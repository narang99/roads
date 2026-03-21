from pathlib import Path
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import string
import json
import cv2
import matplotlib.colors as mcolors
from mtrain.disk import DiskImage, DiskBooleanMask

colors = ["red", "black", "green"]
rd_bk_gn = mcolors.LinearSegmentedColormap.from_list("RdBkGn", ["red", "black", "green"])


def mkdir(p: Path | str):
    Path(p).mkdir(exist_ok=True, parents=True)
    return p


def globL(p: Path | str, pat: str) -> list[Path]:
    return list(Path(p).glob(pat))

def rglobL(p: Path | str, pat: str) -> list[Path]:
    return list(Path(p).rglob(pat))

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

def _plot(crops, figsize=None, ncols=2, axis="on", cmap=None):
    crops = list(crops)
    rows = math.ceil(len(crops) / ncols)
    if figsize is None:
        figsize = (10 * rows, 10 * rows)
    fig, axs = plt.subplots(rows, ncols, figsize=figsize)
    if len(crops) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    for i, c in enumerate(crops):
        if cmap is not None:
            axs[i].imshow(c, cmap=cmap)
        else:
            axs[i].imshow(c)
        axs[i].axis(axis)
    plt.tight_layout()
    return fig, axs


def show(crops, figsize=None, ncols=2, axis="on", cmap=None):
    fig, axs = _plot(crops, figsize, ncols, axis, cmap)
    plt.show()


def save(crops, output_path, figsize=None, ncols=2, axis="on", cmap=None):
    fig, axs = _plot(crops, figsize, ncols, axis, cmap)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


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

def show_with_custom_limit(
    crops, figsize=None, ncols=2, axis="on", cmap=None, limit_getter=None,
):
    crops = list(crops)
    rows = math.ceil(len(crops) / ncols)
    if figsize is None:
        figsize = (5 * rows, 5 * rows)
        print("figsize", figsize)
    _, axs = plt.subplots(rows, ncols, figsize=figsize)
    if len(crops) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    for i, c in enumerate(crops):
        params = {}
        if cmap is not None:
            params["cmap"] = cmap
        if limit_getter is not None:
            vmin, vmax = limit_getter(c)
            params["vmin"] = vmin
            params["vmax"] = vmax

        axs[i].imshow(c, **params)
        axs[i].axis(axis)
    plt.tight_layout()
    plt.show()

def get_local_image_limits(img):
    # return (img.min(), img.max())
    mx, mn = img.max(), img.min()
    mx = max(abs(mx), abs(mn))
    lim = (-mx, mx)
    return lim


def _plot_single_channel_red_green_black(images, figsize=None, ncols=2, axis="on", viztype="global"):
    if not images:
        return None, None, None
    
    if images[0].dtype == np.uint8:
        images = [img.astype(np.float32) for img in images]
    
    all_min = min(img.min() for img in images)
    all_max = max(img.max() for img in images)
    v_limit = max(abs(all_min), abs(all_max))
    
    images = list(images)
    rows = math.ceil(len(images) / ncols)
    if figsize is None:
        figsize = (5 * rows, 5 * rows)
    
    fig, axs = plt.subplots(rows, ncols, figsize=figsize)
    if len(images) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    
    for i, img in enumerate(images):
        params = {}
        if viztype == "gray":
            params["cmap"] = "gray"
        else:
            params["cmap"] = rd_bk_gn
            if viztype == "global":
                params["vmin"], params["vmax"] = -v_limit, v_limit
            elif viztype == "local":
                params["vmin"], params["vmax"] = get_local_image_limits(img)
            else:
                raise Exception(f"invalid viztype {viztype}")
        
        axs[i].imshow(img, **params)
        axs[i].axis(axis)
    
    plt.tight_layout()
    return fig, axs, v_limit


def show_single_channel_red_green_black(images, figsize=None, ncols=2, axis="on", viztype="global"):
    if viztype == "gray":
        show(images, figsize=figsize, ncols=ncols, axis=axis, cmap="gray")
        return
    
    fig, axs, v_limit = _plot_single_channel_red_green_black(images, figsize, ncols, axis, viztype)
    if fig is not None:
        plt.show()


def save_single_channel_red_green_black(images, output_path, figsize=None, ncols=2, axis="on", viztype="global"):
    if viztype == "gray":
        save(images, output_path, figsize=figsize, ncols=ncols, axis=axis, cmap="gray")
        return
    
    fig, axs, v_limit = _plot_single_channel_red_green_black(images, figsize, ncols, axis, viztype)
    if fig is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()


def stack_batch(lofarrays):
    import torch
    tns = []
    for arr in lofarrays:
        if isinstance(arr, torch.Tensor):
            tns.append(arr.detach().clone())
        else:
            tns.append(torch.tensor(arr))
    return torch.stack(tns).unsqueeze(0)