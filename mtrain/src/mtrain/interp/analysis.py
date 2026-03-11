from pathlib import Path
from mtrain.utils import show_single_channel_red_green_black
from typing import Literal, TypedDict
import numpy as np
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BBox:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0


def to_activation_id(layer_id):
    layer_id = layer_id.removesuffix(".weight")
    layer_id = layer_id.removesuffix(".bias")
    return layer_id


def to_weight_id(layer_id):
    if layer_id.endswith(".weight") or layer_id.endswith(".bias"):
        raise Exception(f"ID should not end with weight/bias, id={layer_id}")
    return f"{layer_id}.weight"


def to_bias_id(layer_id):
    if layer_id.endswith(".weight") or layer_id.endswith(".bias"):
        raise Exception(f"ID should not end with weight/bias, id={layer_id}")
    return f"{layer_id}.bias"


def list_layers(root_dir):
    """
    List all layer identifiers from a root directory (weights or activations).

    Args:
        root_dir: Path to weights directory or activation directory

    Returns:
        List of layer identifiers (e.g., ['layers.0.0.weight', 'layers.0.1.bias', ...])
    """
    root_path = Path(root_dir)

    # Check if directory exists
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    # Look for .npy files in the hierarchical structure
    npy_files = list(root_path.rglob("*.npy"))

    # Extract layer identifiers from file paths
    layer_ids = []
    for npy_file in npy_files:
        # Get relative path from root
        rel_path = npy_file.relative_to(root_path)
        # Convert path back to layer identifier (reverse of the save process)
        layer_id = str(rel_path.with_suffix(""))  # Remove .npy
        layer_id = layer_id.replace("/", ".")  # Convert path separators back to dots
        layer_ids.append(layer_id)

    # Sort for consistent ordering
    return sorted(layer_ids)


def get_layer_data(root_dir, layer_id, return_type="all"):
    """
    Get weights/activations/metadata for a specific layer by identifier.

    Args:
        root_dir: Path to weights directory or activation directory
        layer_id: Layer identifier (e.g., 'layers.0.0.weight')
        return_type: 'data', 'metadata', or 'all' (default)

    Returns:
        dict with keys:
        - 'data': numpy array of weights/activations (if requested)
        - 'metadata': dict of metadata (if requested)
        - 'layer_id': the layer identifier
        - 'data_type': 'weights' or 'activations' based on directory
    """
    root_path = Path(root_dir)

    # Check if directory exists
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    # Convert layer_id to file path
    layer_path = root_path / layer_id.replace(".", "/")
    data_file = Path(f"{layer_path}.npy")
    metadata_file = Path(f"{layer_path}_metadata.json")

    # Check if files exist
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    result = {
        "layer_id": layer_id,
        "data_type": "weights"
        if "model_summary.txt" in [f.name for f in root_path.iterdir()]
        else "activations",
    }

    # Load data if requested
    if return_type in ["data", "all"]:
        result["data"] = np.load(data_file)

    # Load metadata if requested
    if return_type in ["metadata", "all"]:
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                # Convert shape back to tuple for consistency
                if "shape" in metadata:
                    metadata["shape"] = tuple(metadata["shape"])
        result["metadata"] = metadata

    return result


def find_weights_discrepancies(activations_dir, weights_dir):
    """Given activations and weights, find the layers where we don't find weights or biases

    Layers like ReLU dont have any weights at all. Some Conv layers won't have biases too.
    This helps us simply verify and find layers without them to make sure our understanding is fine
    """
    layer_ids = list_layers(activations_dir)
    weight_layer_ids = list_layers(weights_dir)

    w_weight_layer_ids = [w for w in weight_layer_ids if w.endswith("weight")]
    w_expected_weight_ids = [f"{lid}.weight" for lid in layer_ids]
    weight_in_actual_but_not_expected = set(w_weight_layer_ids) - set(
        w_expected_weight_ids
    )
    weight_expected_but_not_in_actual = set(w_expected_weight_ids) - set(
        w_weight_layer_ids
    )

    bias_weight_layer_ids = [w for w in weight_layer_ids if w.endswith("bias")]
    bias_expected_weight_ids = [f"{lid}.bias" for lid in layer_ids]
    bias_in_actual_but_not_expected = set(bias_weight_layer_ids) - set(
        bias_expected_weight_ids
    )
    bias_expected_but_not_in_actual = set(bias_expected_weight_ids) - set(
        bias_weight_layer_ids
    )

    return {
        "weights": {
            "in_actual_but_not_expected": list(weight_in_actual_but_not_expected),
            "expected_but_not_in_actual": list(weight_expected_but_not_in_actual),
        },
        "bias": {
            "in_actual_but_not_expected": list(bias_in_actual_but_not_expected),
            "expected_but_not_in_actual": list(bias_expected_but_not_in_actual),
        },
    }


def get_layer_types_of_discreps(layer_ids, activations_dir):
    for dis in layer_ids:
        act_id = to_activation_id(dis)
        data = get_layer_data(activations_dir, act_id)
        yield data["metadata"]["layer_type"]


def print_model_weights_bias_stats(activations_dir, weights_dir):
    discreps = find_weights_discrepancies(activations_dir, weights_dir)
    layer_types_without_weights = set(
        get_layer_types_of_discreps(
            discreps["weights"]["expected_but_not_in_actual"], activations_dir
        )
    )
    layer_types_without_bias = set(
        get_layer_types_of_discreps(
            discreps["bias"]["expected_but_not_in_actual"], activations_dir
        )
    )

    print(
        "layers with weights but no bais",
        layer_types_without_bias - layer_types_without_weights,
    )
    print("layers without weights and bais", layer_types_without_weights)


def get_weights_and_acts(weights_dir, activations_dir):
    """Given a weights and activations directories of the same model with some input, this function yields pairs of activations and corresponding weights in order

    Each weight has an output activation, this model would return the first weight and its activation, second weight and activation, and so on
    If the layer does not have a weight or activation, that part is marked None
    """
    activation_layer_ids = list_layers(activations_dir)
    weights_layer_ids = set(list_layers(weights_dir))
    for layer_id in activation_layer_ids:
        w_layer_id = to_weight_id(layer_id)
        b_layer_id = to_bias_id(layer_id)

        activation = get_layer_data(activations_dir, layer_id)
        weight, bias = None, None
        if b_layer_id in weights_layer_ids:
            bias = get_layer_data(weights_dir, b_layer_id)
        if w_layer_id in weights_layer_ids:
            weight = get_layer_data(weights_dir, w_layer_id)
        yield {
            "activation": activation,
            "weight": weight,
            "bias": bias,
            "layer_id": layer_id,
        }


def get_per_channel_conv(layer_id, input_batch, model, kernel_idx=0):
    # 1. Get the layer and its properties
    layer = model.get_submodule(layer_id)
    device = next(layer.parameters()).device
    weights = layer.weight[kernel_idx]
    single_kernel = weights.unsqueeze(1)
    input_batch = input_batch.to(device)
    in_channels = input_batch.shape[1]
    with torch.no_grad():
        per_channel_output = F.conv2d(
            input_batch,
            single_kernel,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,  # Pass dilation here
            groups=in_channels,  # Keep channels separate
        )

    return per_channel_output


def normalize(img_tensor, mean, std):
    """
    Normalizes a tensor [C, H, W] using per-channel mean and std.
    """
    device = img_tensor.device
    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    if img_tensor.dtype == torch.uint8 or img_tensor.max() > 1.0:
        print("max is great hahahaha", img_tensor.max())
        img_tensor = img_tensor.float() / 255.0
    return (img_tensor - mean) / std


def get_input_patch_padded(input_batch, layer_id, model, out_pos):
    """
    Returns the exact input patch [B, C, K_eff_H, K_eff_W] that produced
    the output at out_pos, padding with zeros where the kernel went out of bounds.
    """
    y, x = out_pos
    B, C, H_in, W_in = input_batch.shape

    layer = model.get_submodule(layer_id)

    # 1. Extract layer parameters
    stride = (
        layer.stride
        if isinstance(layer.stride, tuple)
        else (layer.stride, layer.stride)
    )
    padding = (
        layer.padding
        if isinstance(layer.padding, tuple)
        else (layer.padding, layer.padding)
    )
    dilation = (
        layer.dilation
        if isinstance(layer.dilation, tuple)
        else (layer.dilation, layer.dilation)
    )
    k_h, k_w = layer.kernel_size

    # 2. Calculate effective kernel size
    k_eff_h = (k_h - 1) * dilation[0] + 1
    k_eff_w = (k_w - 1) * dilation[1] + 1

    # 3. Calculate theoretical input boundaries (can be negative!)
    in_y_start = y * stride[0] - padding[0]
    in_x_start = x * stride[1] - padding[1]
    in_y_end = in_y_start + k_eff_h
    in_x_end = in_x_start + k_eff_w

    # 4. Calculate how much we need to crop vs. how much we need to pad
    # We find the intersection of the kernel and the actual image
    slice_y_start = max(0, in_y_start)
    slice_x_start = max(0, in_x_start)
    slice_y_end = min(H_in, in_y_end)
    slice_x_end = min(W_in, in_x_end)

    # Extract the valid part of the image
    patch = input_batch[:, :, slice_y_start:slice_y_end, slice_x_start:slice_x_end]

    # 5. Calculate padding amounts for the patch
    # pad syntax: (left, right, top, bottom)
    pad_left = max(0, -in_x_start)
    pad_right = max(0, in_x_end - W_in)
    pad_top = max(0, -in_y_start)
    pad_bottom = max(0, in_y_end - H_in)

    # Apply padding to the extracted patch to restore it to [K_eff_H, K_eff_W]
    padded_patch = F.pad(
        patch, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    return padded_patch, BBox(slice_y_start, slice_x_start, slice_y_end, slice_x_end)


def get_input_bbox_padded(
    input_batch: torch.Tensor, layer_id: str, model, bbox_out: BBox
) -> tuple[torch.Tensor, BBox]:
    """
    input_batch: [B, C, H, W]
    layer: The nn.Conv2d layer
    bbox_out: A BBox object representing indices in the output feature map
    """
    B, C, H_in, W_in = input_batch.shape
    layer = model.get_submodule(layer_id)

    # 1. Extract layer parameters
    stride = (
        layer.stride
        if isinstance(layer.stride, tuple)
        else (layer.stride, layer.stride)
    )
    padding = (
        layer.padding
        if isinstance(layer.padding, tuple)
        else (int(layer.padding), int(layer.padding))
    )
    dilation = (
        layer.dilation
        if isinstance(layer.dilation, tuple)
        else (layer.dilation, layer.dilation)
    )
    k_h, k_w = layer.kernel_size

    # 2. Calculate effective kernel size
    k_eff_h = (k_h - 1) * dilation[0] + 1
    k_eff_w = (k_w - 1) * dilation[1] + 1
    # 3. Calculate theoretical input boundaries using BBox attributes
    # The start is determined by the top-left pixel's top-left corner
    in_y_start = bbox_out.y0 * stride[0] - padding[0]
    in_x_start = bbox_out.x0 * stride[1] - padding[1]

    # The end is determined by the bottom-right pixel's bottom-right corner
    # (Assuming y_max/x_max are inclusive indices)
    in_y_end = (bbox_out.y1 * stride[0] - padding[0]) + k_eff_h
    in_x_end = (bbox_out.x1 * stride[1] - padding[1]) + k_eff_w

    # 4. Determine valid slice boundaries
    slice_y_start = max(0, in_y_start)
    slice_x_start = max(0, in_x_start)
    slice_y_end = min(H_in, in_y_end)
    slice_x_end = min(W_in, in_x_end)

    # 5. Extract the valid part of the image
    if slice_y_start >= slice_y_end or slice_x_start >= slice_x_end:
        # Return an empty/zero tensor of the expected theoretical size
        return torch.zeros((B, C, in_y_end - in_y_start, in_x_end - in_x_start)), BBox(
            in_y_start, in_x_start, in_y_end, in_x_end
        )

    patch = input_batch[:, :, slice_y_start:slice_y_end, slice_x_start:slice_x_end]

    # 6. Calculate required padding
    pad_left = max(0, -in_x_start)
    pad_right = max(0, in_x_end - W_in)
    pad_top = max(0, -in_y_start)
    pad_bottom = max(0, in_y_end - H_in)

    # 7. Apply padding
    padded_patch = F.pad(
        patch, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )
    bbox = BBox(slice_y_start, slice_x_start, slice_y_end, slice_x_end)
    return padded_patch, bbox


def get_visual_kernel(layer, kernel_idx=0):
    """
    Returns an inflated 3D kernel [In_Channels, H_eff, W_eff]
    accounting for dilation by inserting zeros.
    """
    # 1. Extract weights for the specific filter: [In_Channels, K_H, K_W]
    # We use .detach().cpu() to ensure it's plottable/manipulatable
    weights = layer.weight[kernel_idx].detach().cpu()
    in_channels, k_h, k_w = weights.shape

    # 2. Extract dilation (handle both int and tuple)
    dilation = layer.dilation
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    dil_h, dil_w = dilation

    # 3. Calculate the inflated (visual) dimensions
    h_eff = (k_h - 1) * dil_h + 1
    w_eff = (k_w - 1) * dil_w + 1

    # 4. Create an empty (zero) tensor for the 3D volume
    visual_kernel_3d = torch.zeros((in_channels, h_eff, w_eff))

    # 5. Populate the inflated tensor
    # The [:, ::dil_h, ::dil_w] slice selects every 'dilation-th' pixel
    # across all input channels simultaneously.
    visual_kernel_3d[:, ::dil_h, ::dil_w] = weights

    return visual_kernel_3d


class RunResult(TypedDict):
    input: list[np.ndarray]
    per_chan: list[np.ndarray]
    kernel: list[np.ndarray]
    output: np.ndarray


def run_and_show_for_input(
    layer_id,
    input_tensor,
    model,
    kernel_idx,
    figsize=None,
    viztype: Literal["global", "local", "gray"] = "global",
) -> RunResult:
    layer = model.get_submodule(layer_id)
    kernel = layer.weight[kernel_idx].detach().numpy()
    o0_batch = run_on_layer(layer_id, input_tensor, model)
    o0 = o0_batch[0][kernel_idx]
    per_chan_conv = get_per_channel_conv(layer_id, input_tensor, model, kernel_idx)[0]

    manual_sum = per_chan_conv.sum(dim=0)
    are_equal = torch.allclose(o0, manual_sum, atol=1e-6)
    if not are_equal:
        raise Exception("manual sum not equal to actual layer output")
    o0 = o0.numpy()

    to_show_input = [w for w in input_tensor[0].numpy()]
    ktoshow = [k for k in kernel]
    if len(ktoshow) != len(to_show_input):
        raise Exception(
            f"WARN: kernel length is not the same as the input length, kernel-length={len(ktoshow)} input-length={len(to_show_input)}"
        )
    ncols = len(to_show_input)
    per_chan_to_show = [c for c in per_chan_conv.numpy()]
    to_show = to_show_input + per_chan_to_show + ktoshow + [o0]
    show_single_channel_red_green_black(
        to_show, figsize=figsize, ncols=ncols, viztype=viztype
    )

    return {
        "input": to_show_input,
        "per_chan": per_chan_to_show,
        "kernel": ktoshow,
        "output": o0,
    }


def run_on_layer(layer_id, input_batch, model):
    layer = model.get_submodule(layer_id)
    device = next(layer.parameters()).device
    input_batch = input_batch.to(device)
    with torch.no_grad():
        return layer(input_batch)


def print_conv_stats(res: RunResult):
    # kernel wide stats
    raw_kernel_sums = [k.sum() for k in res["kernel"]]
    abs_kernel_sums = [np.abs(k).sum() for k in res["kernel"]]
    raw_per_chan_sums = [p.sum() for p in res["per_chan"]]
    abs_per_chan_sums = [np.abs(p).sum() for p in res["per_chan"]]

    print("\nRaw Kernel sums:")
    for i, s in enumerate(raw_kernel_sums):
        print("\t", i, s)
    print("total", sum(raw_kernel_sums))

    print("\nAbsolute Kernel sums:")
    for i, s in enumerate(abs_kernel_sums):
        print("\t", i, s)
    print("total", sum(abs_kernel_sums))

    print("\nRaw Per channel sums:")
    for i, s in enumerate(raw_per_chan_sums):
        print("\t", i, s)
    print("total", sum(raw_per_chan_sums))

    print("\nAbs Per channel sums:")
    for i, s in enumerate(abs_per_chan_sums):
        print("\t", i, s)
    print("total", sum(abs_per_chan_sums))

    print("\nOutput raw sum", res["output"].sum())
    print("\nOutput abs sum", np.abs(res["output"]).sum())
