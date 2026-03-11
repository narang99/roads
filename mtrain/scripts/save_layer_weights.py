from pathlib import Path
import numpy as np
import torch
import argparse
import sys
from mtrain.interp.save import (
    save_layer_weights,
    save_model_summary,
    ActivationCapture,
    save_layer_activations,
)
from fastai.vision.all import load_learner, PILImage

# Default paths - will be overridden by CLI args
DEFAULT_MODEL_PATH = "/Users/hariomnarang/Desktop/gdrive-sync/garbage/experiments/enguled-bbox-levels-crops-v3/log/export_iter_14.pkl"
DEFAULT_WEIGHTS_DIR = "./weights"


def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return load_learner(model_path)


##################################################### learner100.model ################################################
##################### this is a resnet18 model created using unet_learner in fastai ###################################
## we have the architecture here
# Sequential(
#   (0): Sequential(
#     (0): Conv2d(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#     (4): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (5): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (downsample): Sequential(
#           (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (6): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (downsample): Sequential(
#           (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (7): Sequential(
#       (0): BasicBlock(
#         (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (downsample): Sequential(
#           (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): BasicBlock(
#         (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#   )
#   (1): Sequential(
#     (0): AdaptiveConcatPool2d(
#       (ap): AdaptiveAvgPool2d(output_size=1)
#       (mp): AdaptiveMaxPool2d(output_size=1)
#     )
#     (1): fastai.layers.Flatten(full=False)
#     (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (3): Dropout(p=0.25, inplace=False)
#     (4): Linear(in_features=1024, out_features=512, bias=False)
#     (5): ReLU(inplace=True)
#     (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (7): Dropout(p=0.5, inplace=False)
#     (8): Linear(in_features=512, out_features=2, bias=False)
#   )
# )
#############################################################################################################

############## code session start ############33
##### write a scriopt to get all intermediate activations from this model and save them in a directory


def process_image_and_save_activations(
    image_path, activation_dir, weights_dir, model_path=None
):
    """
    Process a single image and save both layer weights and activations.

    Args:
        image_path: Path to input image
        activation_dir: Directory to save activations (named after image)
        weights_dir: Directory to save weights (optional, defaults to DEFAULT_WEIGHTS_DIR)
        model_path: Path to model file (optional)
    """
    # Load model
    print(f"Loading model from {model_path or DEFAULT_MODEL_PATH}...")
    learner = load_model(model_path)

    # Load and process image
    print(f"Processing image: {image_path}")
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image using fastai PILImage
    img = PILImage.create(img_path)

    # Setup activation capture
    activation_capture = ActivationCapture()
    activation_capture.register_hooks(learner.model)

    # Run inference to capture activations
    print("Running inference and capturing activations...")
    with torch.no_grad():
        prediction = learner.predict(img)

    # Save weights to specified directory
    weights_path = Path(weights_dir)
    print(f"Saving layer weights to {weights_path}...")
    save_layer_weights(learner.model, weights_path)
    save_model_summary(learner.model, weights_path)

    # Save activations to image-specific directory
    activations_dir = Path(activation_dir)
    print(f"Saving layer activations to {activations_dir}...")
    save_layer_activations(activation_capture.activations, activation_capture.layer_types, activations_dir)

    # Clean up hooks
    activation_capture.clear_hooks()

    # Save inference result info in both directories
    inference_info = f"""Input image: {image_path}
Image shape: {img.size}
Prediction: {prediction[0]}
Confidence: {float(prediction[2].max()):.4f}
"""

    with open(weights_path / "inference_info.txt", "w") as f:
        f.write(inference_info)

    with open(activations_dir / "inference_info.txt", "w") as f:
        f.write(inference_info)

    print(f"✓ Layer weights saved to {weights_path}")
    print(f"✓ Layer activations saved to {activations_dir}")

    try:
        save_raw_input(learner, image_path, activations_dir)
    except Exception as ex:
        print(f"WARN: failure in saving raw input, path={image_path} cause={ex}")


def validate_activation_directory(activation_dir):
    activation_path = Path(activation_dir)

    if activation_path.exists():
        print(
            f"Error: Activation directory already exists: {activation_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create parent directory if it doesn't exist
    activation_path.parent.mkdir(parents=True, exist_ok=True)


def validate_weights_directory(weights_dir):
    weights_path = Path(weights_dir)
    if weights_path.exists():
        print(
            f"Warning: Weights directory exists and will be overwritten: {weights_path}"
        )
    else:
        print(f"Weights will be saved to: {weights_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract and save layer weights and activations from neural network inference on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python save_layer_weights.py image.jpg
  python save_layer_weights.py image.jpg --output-dir ./image_activations
  python save_layer_weights.py image.jpg -o ./custom_activations --model ./my_model.pkl

Notes:
  - Weights are saved to --weights-dir or ./weights by default (overwritten if exists)
  - Activations are saved to specified output directory or ./<image_name> if not specified
        """,
    )

    parser.add_argument("image", help="Path to input image file")

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for activations (default: ./<image_name>)",
    )

    parser.add_argument(
        "--model", help=f"Path to model file (default: {DEFAULT_MODEL_PATH})"
    )

    parser.add_argument(
        "--weights-dir",
        help=f"Directory to save weights (default: {DEFAULT_WEIGHTS_DIR})",
        default=str(DEFAULT_WEIGHTS_DIR),
    )

    return parser.parse_args()


def save_raw_input(learner, image_path, out_activations_dir):
    # 1. Load your image
    dl = learner.dls.test_dl([image_path])
    ip_arr = dl.one_batch()[0].cpu().numpy()
    dest_dir = Path(out_activations_dir) / "__model_input__"
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.save(dest_dir / "input.npy", ip_arr)


def main():
    """Main CLI entry point"""
    args = parse_arguments()

    # Determine activation output directory
    if args.output_dir:
        activation_dir = args.output_dir
    else:
        # Use image name without extension
        image_path = Path(args.image)
        image_name = image_path.stem
        activation_dir = f"./{image_name}"

    # Validate directories
    validate_weights_directory(args.weights_dir)
    validate_activation_directory(activation_dir)

    try:
        # Process image and save weights + activations
        process_image_and_save_activations(
            image_path=args.image,
            activation_dir=activation_dir,
            weights_dir=args.weights_dir,
            model_path=args.model,
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
