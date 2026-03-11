from pathlib import Path
import numpy as np
import json
from mtrain.utils import mkdir


def save_layer_weights(model, save_dir):
    save_path = Path(save_dir)
    mkdir(save_path)

    for module_name, module in model.named_modules():
        module_full_name = type(module).__name__
        print(module_full_name)
        for name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{name}" if module_name else name
            print("\t", name)
            if param.requires_grad:
                # Create subdirectory structure based on layer name
                layer_path = save_path / name.replace(".", "/")
                layer_path.parent.mkdir(parents=True, exist_ok=True)

                # Save weight tensor as numpy array
                weight_data = param.detach().cpu().numpy()
                np.save(f"{layer_path}.npy", weight_data)

                # Determine layer type
                layer_type = module_full_name
                # layer_type = "unknown"
                # if "conv" in name.lower():
                #     layer_type = "conv"
                # elif "bn" in name.lower() or "batchnorm" in name.lower():
                #     layer_type = "batch_norm"
                # elif "linear" in name.lower() or "fc" in name.lower():
                #     layer_type = "linear"
                # else:
                #     print("unknown layer type", name)

                # Save metadata with layer type information
                metadata = {
                    "layer_name": name,
                    "layer_type": layer_type,
                    "shape": list(
                        weight_data.shape
                    ),  # Convert to list for JSON serialization
                    "dtype": str(weight_data.dtype),
                    "requires_grad": param.requires_grad,
                    "device": str(param.device),
                    "num_parameters": int(weight_data.size),
                }

                with open(f"{layer_path}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)


def save_model_summary(model, save_dir):
    save_path = Path(save_dir)
    mkdir(save_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(save_path / "model_summary.txt", "w") as f:
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n\n")
        f.write("Model Architecture:\n")
        f.write(str(model))


class ActivationCapture:
    def __init__(self):
        self.activations = {}
        self.layer_types = {}
        self.hooks = []

    def hook_fn(self, name):
        def hook(module, input, output):
            # If output is a tuple (some layers do this), grab the first tensor
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()
        return hook

    def register_hooks(self, model):
        self.clear_hooks()
        for name, module in model.named_modules():
            # Only leaf modules (actual layers like Conv2d, Linear, etc.)
            if len(list(module.children())) == 0:
                self.layer_types[name] = type(module).__name__
                
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.layer_types = {}

def save_layer_activations(activations, layer_types, save_dir):
    save_path = Path(save_dir)
    mkdir(save_path)

    for name, activation in activations.items():
        if activation is not None:
            # Create subdirectory structure based on layer name
            layer_path = save_path / name.replace(".", "/")
            layer_path.parent.mkdir(parents=True, exist_ok=True)

            # Save activation tensor as numpy array
            activation_data = activation.numpy()
            np.save(f"{layer_path}.npy", activation_data)

            # Determine layer type
            # layer_type = "unknown"
            # if "conv" in name.lower():
            #     layer_type = "conv"
            # elif "bn" in name.lower() or "batchnorm" in name.lower():
            #     layer_type = "batch_norm"
            # elif "linear" in name.lower() or "fc" in name.lower():
            #     layer_type = "linear"
            layer_type = layer_types.get(name, "unknown")

            # Save metadata with layer type information
            metadata = {
                "layer_name": name,
                "layer_type": layer_type,
                "shape": list(
                    activation_data.shape
                ),  # Convert to list for JSON serialization
                "dtype": str(activation_data.dtype),
                "min_value": float(activation_data.min()),
                "max_value": float(activation_data.max()),
                "mean_value": float(activation_data.mean()),
                "std_value": float(activation_data.std()),
                "num_elements": int(activation_data.size),
            }

            with open(f"{layer_path}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
