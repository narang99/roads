import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mtrain.disk import DiskBooleanMask, DiskImage
from mtrain.example_dir.core import ExampleDir, load_npz
from mtrain.example_dir.defaults.negmask import default_negmask_learners
from mtrain.example_dir.defaults.smallnet import default_smallnet_learners
from mtrain.example_dir.iterdir import get_dirs

# Constants - adjusted to match your environment
TEST_SET_PATH = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/inference/test_set")
MODELS_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/models")
REPORT_ROOT = Path("reports")

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def create_diff_image(image, mask_baseline, mask_new):
    # Dim the original image for background
    bg = (image.astype(float) * 0.4).astype(np.uint8)
    
    # Agreement: White
    agreement = np.logical_and(mask_baseline, mask_new)
    bg[agreement] = [255, 255, 255]
    
    # Expansion: Green (New has it, Baseline doesn't)
    expansion = np.logical_and(mask_new, ~mask_baseline)
    bg[expansion] = [0, 255, 0]
    
    # Contraction: Red (Baseline has it, New doesn't)
    contraction = np.logical_and(mask_baseline, ~mask_new)
    bg[contraction] = [255, 0, 0]
    
    return bg

def main():
    parser = argparse.ArgumentParser(description="Evaluate a new negmask iteration against baseline.")
    parser.add_argument("--label", required=True, help="Label of the new model iteration.")
    parser.add_argument("--force-new-model", help="Force predictions of the new model.", action="store_true")
    parser.add_argument("--baseline", default="baseline", help="Label of the baseline model.")
    parser.add_argument("--bs", type=int, default=8, help="Batch size for learners.")
    parser.add_argument("--test-dir", default=TEST_SET_PATH, type=Path, help="Location of the directory containing examples (of structure ExampleDir uses)")
    args = parser.parse_args()

    new_label = args.label
    baseline_label = args.baseline
    
    report_dir = REPORT_ROOT / f"eval_{new_label}_vs_{baseline_label}"
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading learners for labels: {baseline_label}, {new_label}...")
    smallnet = default_smallnet_learners(MODELS_DIR, ["md"], args.bs)
    negmask = default_negmask_learners(MODELS_DIR, [baseline_label, new_label], args.bs)

    test_dir = Path(args.test_dir)
    dirs = list(get_dirs(test_dir))
    print(f"Found {len(dirs)} directories in test set.")
    print("Will force new model? ", args.force_new_model)

    results = []

    for d in tqdm(dirs, desc="Evaluating"):
        try:
            edir = ExampleDir(d, smallnet, negmask)
            
            # 1. Ensure predictions exist
            edir.negmask_paths(baseline_label, "md", force=False)
            edir.negmask_paths(new_label, "md", force=args.force_new_model)
            
            # 2. Get masks
            m_base = edir.get_trash_mask(baseline_label, "md")
            m_new = edir.get_trash_mask(new_label, "md")
            
            # 3. Calculate Metrics
            iou = calculate_iou(m_base, m_new)
            
            trash_pixels_base = m_base.sum()
            trash_pixels_new = m_new.sum()
            
            # Flips
            to_trash = np.logical_and(m_new, ~m_base).sum()
            to_other = np.logical_and(m_base, ~m_new).sum()
            
            # 4. Generate Visuals
            image = edir.load_and_resize_image(edir.image_path)
            diff_img = create_diff_image(image, m_base, m_new)
            
            # Save assets
            img_name = f"{d.name}_diff.jpg"
            DiskImage.save(diff_img, assets_dir / img_name)
            
            results.append({
                "id": d.name,
                "iou": iou,
                "trash_base": int(trash_pixels_base),
                "trash_new": int(trash_pixels_new),
                "to_trash": int(to_trash),
                "to_other": int(to_other),
                "img_path": f"assets/{img_name}"
            })
            
        except Exception as e:
            print(f"\nError processing {d}: {e}")

    # Sort by IoU ascending (most different first)
    results.sort(key=lambda x: x["iou"])

    # 5. Generate HTML
    html_content = f"""
    <html>
    <head>
        <title>Evaluation: {new_label} vs {baseline_label}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #f0f0f0; }}
            table {{ border-collapse: collapse; width: 100%; background: white; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .diff-img {{ width: 400px; cursor: pointer; }}
            .legend {{ margin-bottom: 20px; padding: 10px; background: white; border-radius: 5px; }}
            .swatch {{ display: inline-block; width: 20px; height: 20px; vertical-align: middle; margin-right: 5px; border: 1px solid #000; }}
        </style>
    </head>
    <body>
        <h1>Negmask Evaluation:  {new_label} vs {baseline_label} (New vs Old)</h1>
        <div class="legend">
            <strong>Legend:</strong><br>
            <span class="swatch" style="background: white;"></span> Agreement (Both Trash)<br>
            <span class="swatch" style="background: #00ff00;"></span> Expansion (Iteration Only)<br>
            <span class="swatch" style="background: #ff0000;"></span> Contraction (Baseline Only)<br>
            <em>Images are sorted by IoU (Most different at the top).</em>
        </div>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>IoU</th>
                    <th>Base Trash Px</th>
                    <th>New Trash Px</th>
                    <th>Flipped to Trash</th>
                    <th>Flipped to Other</th>
                    <th>Visual Diff</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for r in results:
        html_content += f"""
                <tr>
                    <td>{r['id']}</td>
                    <td>{r['iou']:.4f}</td>
                    <td>{r['trash_base']}</td>
                    <td>{r['trash_new']}</td>
                    <td>{r['to_trash']}</td>
                    <td>{r['to_other']}</td>
                    <td><a href="{r['img_path']}" target="_blank"><img class="diff-img" src="{r['img_path']}"></a></td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(report_dir / "index.html", "w") as f:
        f.write(html_content)
        
    print(f"\nReport generated at: {report_dir.absolute()}/index.html")

if __name__ == "__main__":
    main()
