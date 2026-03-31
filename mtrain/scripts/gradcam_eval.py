import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from fastai.vision.all import (
    load_learner,
    vision_learner,
    xresnet18,
    DataLoaders,
    CrossEntropyLossFlat,
    F1Score,
    Precision,
    Recall,
    default_device,
)
from mtrain.neg_mask.model.datasets.blur_pad_dl import BlurPadDataset, random_tfm, BlurPad4ChanDataset
from mtrain.denorm import denormalize_imagenet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from functools import partial
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import functional as F

def get_denormalized(input_tensor):
    if input_tensor.shape[0] == 4:
        input_tensor = input_tensor[:3, :, :]
    img_denorm = denormalize_imagenet(input_tensor).permute(1, 2, 0).cpu().numpy()
    return img_denorm

class CustomN(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.body = encoder
        self.head = head
        self.mask_weight = nn.Parameter(torch.full((1, 512, 1, 1), 0.5))

    def forward(self, x):
        images = x[:,:3,:,:]
        masks = x[:,3:4,:,:]
        activs = self.body(images)
        pooled = F.adaptive_max_pool2d(masks, (7, 7))
        m = (pooled * (1.0 - self.mask_weight)) + self.mask_weight
        constrained_activs = activs * m
        out = self.head(constrained_activs)
        return out

# class CustomN(nn.Module):
#     def __init__(self, encoder, head):
#         super().__init__()
#         self.body = encoder
#         self.head = head

#     def forward(self, x):
#         # 1. Pass through ResNet body (CNN layers)
#         # x is [b, 4, h, w]
#         # we push [b, 3, h, w] to the body
#         images = x[:, :3, :, :]
#         masks = x[:, 3:4, :, :]

#         activs = self.body(images)

#         pooled = F.adaptive_max_pool2d(masks, (7, 7))
#         # should this 0.5 be a learnable parameter?
#         pooled = (pooled * 0.5) + 0.5

#         constrained_activs = activs * pooled
#         # 2. Apply your custom function
#         # This must return a tensor of the same shape as the body output
#         # x = self.custom_func(x)

#         # 3. Pass through fastai head (Pooling + Linear layers)
#         out = self.head(constrained_activs)
#         return out


def get_dls(ds_path, num_samples, crop_size=224, dataset_class:str = "BlurPadDataset"):
    image_paths = list((Path(ds_path) / "train").glob("*.jpg"))[:num_samples]

    def rm_mapi_walls(path):
        if "mapillary" in path.name:
            return False
        else:
            return True

    image_paths = list(filter(rm_mapi_walls, image_paths))

    # Use a fixed transform for validation evaluation
    def eval_tfm(img, mask, bbox):
        from mtrain.neg_mask.model.datasets.blur_pad_dl import CropTfmsOutsideBbox

        return CropTfmsOutsideBbox(img, bbox).step_down(0.5).crop, mask

    ds_class_to_cls = {
        "BlurPadDataset": BlurPadDataset,
        "BlurPad4ChanDataset": BlurPad4ChanDataset,
    }
    dataset_cls = ds_class_to_cls[dataset_class]
    valid_ds = dataset_cls(
        image_paths,
        Path(ds_path) / "masks",
        crop_size,
        True,
        crop_mutator=eval_tfm,
        bbox_pad=3,
    )
    return valid_ds


def get_learner(model_path, device, arch: str):
    dls = DataLoaders.from_dsets([], [])  # dummy
    learn = vision_learner(
        dls,
        xresnet18,
        n_out=2,
        pretrained=True,
        normalize=False,
        loss_func=CrossEntropyLossFlat(),
    )
    # change model for CustomNN
    if arch == "CustomN":
        body = learn.model[0]
        head = learn.model[1]
        model = CustomN(body, head)
        learn.model = model
        learn.dls = dls
        learn.n_in = 4
        learn.splitter = lambda m: [m.body, m.head]

    state_dict = torch.load(model_path, map_location=device)
    learn.model.load_state_dict(state_dict)
    learn.model.to(device)
    learn.model.eval()
    return learn


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / union


def run_gradcam_eval(
    model_path,
    ds_path,
    output_dir,
    num_samples=100,
    layer_name="0.7.1.convpath.1.0",
    arch="xresnet18",
    dataset_type="BlurPadDataset",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    device = default_device()

    learn = get_learner(model_path, device, arch)

    # Load dataset
    valid_ds = get_dls(ds_path, num_samples, dataset_class=dataset_type)
    print("haha", len(valid_ds), valid_ds)

    target_layer = [learn.model.get_submodule(layer_name)]
    cam = GradCAM(model=learn.model, target_layers=target_layer)

    results = []

    print(f"Running evaluation on {len(valid_ds)} samples...")
    for i in tqdm(range(len(valid_ds))):
        input_tensor, target = valid_ds[i]
        # Use the direct method to get the correct aligned mask
        _, mask_tensor = valid_ds.get_image_and_mask_tensor(i)

        input_batch = input_tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = learn.model(input_batch)
            pred = output.argmax(dim=1).item()
            prob = F.softmax(output, dim=1)[0, pred].item()

        # Generate CAM
        targets = [ClassifierOutputTarget(target.item())]
        grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0, :]

        # Process images for report
        # print("shape", input_tensor.shape)
        img_denorm = get_denormalized(input_tensor)
        # img_denorm = denormalize_imagenet(input_tensor).permute(1, 2, 0).cpu().numpy()
        cam_image = show_cam_on_image(img_denorm, grayscale_cam, use_rgb=True)

        # Binary mask from dataset method
        binary_mask = (mask_tensor.squeeze().cpu().numpy() > 0).astype(np.float32)

        # Compare CAM with mask
        # Threshold CAM to get a binary focus area
        cam_binary = (grayscale_cam > 0.5).astype(np.float32)
        iou = calculate_iou(binary_mask, cam_binary)

        # Save visuals
        image_name = valid_ds.image_paths[i].name
        cam_name = f"cam_{i}.jpg"
        mask_name = f"mask_{i}.jpg"
        cv2.imwrite(
            str(assets_dir / cam_name), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(assets_dir / mask_name), (binary_mask * 255).astype(np.uint8))

        results.append(
            {
                "idx": i,
                "path": image_name,
                "target": target.item(),
                "pred": pred,
                "prob": prob,
                "iou": iou,
                "cam_path": f"assets/{cam_name}",
                "mask_path": f"assets/{mask_name}",
            }
        )

    # Sort by IoU ascending (worst focus match first)
    results.sort(key=lambda x: x["iou"])

    ious = [r["iou"] for r in results]
    max_iou = max(ious) if ious else 0
    min_iou = min(ious) if ious else 0
    avg_iou = sum(ious) / len(ious) if ious else 0
    non_zero_count = sum(1 for i in ious if i > 0)

    # Generate HTML
    html = f"""
    <html>
    <head>
        <title>Grad-CAM vs Mask Evaluation</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #f0f0f0; }}
            .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stats {{ display: flex; gap: 40px; font-size: 1.2em; }}
            .stat-item {{ display: flex; flex-direction: column; }}
            .stat-label {{ color: #666; font-size: 0.8em; text-transform: uppercase; }}
            .stat-value {{ font-weight: bold; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; background: white; }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: center; }}
            img {{ width: 300px; }}
            .bad {{ background-color: #ffcccc; }}
            .good {{ background-color: #ccffcc; }}
        </style>
    </head>
    <body>
        <h1>Grad-CAM focus vs Ground Truth Mask</h1>
        
        <div class="summary">
            <h3>Focus Performance (IoU)</h3>
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-label">Maximum</span>
                    <span class="stat-value">{max_iou:.4f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Average</span>
                    <span class="stat-value">{avg_iou:.4f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Minimum</span>
                    <span class="stat-value">{min_iou:.4f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Non-Zero IoU</span>
                    <span class="stat-value">{non_zero_count} / {len(results)} ({non_zero_count / len(results) * 100:.1f}%)</span>
                </div>
            </div>
        </div>

        <p>Sorted by IoU (lowest first). Low IoU means model is focusing on wrong areas.</p>
        <table>
            <tr>
                <th>Idx</th>
                <th>Path</th>
                <th>Target</th>
                <th>Pred</th>
                <th>Prob</th>
                <th>IoU (Focus)</th>
                <th>Grad-CAM</th>
                <th>Mask</th>
            </tr>
    """
    for r in results:
        row_class = "good" if r["iou"] > 0 else "bad"
        html += f"""
            <tr class="{row_class}">
                <td>{r["idx"]}</td>
                <td>{r["path"]}</td>
                <td>{r["target"]}</td>
                <td>{r["pred"]}</td>
                <td>{r["prob"]:.2f}</td>
                <td>{r["iou"]:.4f}</td>
                <td><img src="{r["cam_path"]}"></td>
                <td><img src="{r["mask_path"]}"></td>
            </tr>
        """
    html += "</table></body></html>"

    with open(output_dir / "index.html", "w") as f:
        f.write(html)

    print(f"Report generated at {output_dir}/index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth model weights")
    parser.add_argument(
        "--ds",
        required=True,
        help="Path to dataset root (containing 'train' and 'masks')",
    )
    parser.add_argument(
        "--out", default="reports/gradcam_eval", help="Output directory"
    )
    parser.add_argument(
        "-n", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--layer", default="0.7.1.convpath.1.0", help="Layer for Grad-CAM"
    )
    parser.add_argument("--arch", default="xresnet18", help="Model architecture")
    parser.add_argument(
        "--dataset-type", default="BlurPadDataset", help="Dataset class name"
    )

    args = parser.parse_args()

    run_gradcam_eval(
        args.model, args.ds, args.out, args.n, args.layer, args.arch, args.dataset_type
    )


if __name__ == "__main__":
    main()
