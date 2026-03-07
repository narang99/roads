"""
CLIP Image Search
Usage:
    python clip_search.py --dir /path/to/images --query "rocks on road" --out results.txt
    python clip_search.py --dir /path/to/images --query "rocks on road" --out results.txt --top 50 --threshold 0.2
"""

import argparse
import pickle
from pathlib import Path

import torch
import clip
from PIL import Image

from mtrain.utils import mkdir


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# def get_image_paths(directory):
#     directory = Path(directory)
#     return [
#         p for p in directory.rglob("image.jpg")
#         if p.suffix.lower() in SUPPORTED_EXTENSIONS
#     ]

def get_image_paths(directory):
    directory = Path(directory)
    return [
        p for p in directory.glob("*.jpg")
    ]

def load_model(device):
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def embed_images(image_paths, model, preprocess, device, cache_path=None):
    # Load from cache if available
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Embedding {len(image_paths)} images...")
    embeddings = []
    valid_paths = []

    for i, path in enumerate(image_paths):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(emb.cpu())
            valid_paths.append(str(path))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(image_paths)} done...")
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    result = {"embeddings": torch.cat(embeddings), "paths": valid_paths}

    if cache_path:
        print(f"Saving embeddings cache to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result


def search(query, data, model, device, top_k=None, threshold=None):
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    embeddings = data["embeddings"].to(device)
    scores = (embeddings @ text_emb.T).squeeze(1).cpu()

    results = sorted(
        zip(scores.tolist(), data["paths"]),
        key=lambda x: x[0],
        reverse=True
    )

    if threshold is not None:
        results = [(s, p) for s, p in results if s >= threshold]

    if top_k is not None:
        results = results[:top_k]

    return results


def main():
    CACHE_DEFAULT_PATH = str(mkdir(Path.home() / ".roads-clip-cache") / "embeddings.pkl")
    parser = argparse.ArgumentParser(description="Search images using CLIP")
    parser.add_argument("--dir", required=True, help="Directory of images")
    parser.add_argument("--query", required=True, help="Text query, e.g. 'rocks on road'")
    parser.add_argument("--out", required=True, help="Output file for matched image paths")
    parser.add_argument("--top", type=int, default=None, help="Return top N results (default: all above threshold)")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum similarity score (default: 0.2)")
    parser.add_argument("--cache", type=str, default=CACHE_DEFAULT_PATH, help="Path to save/load embedding cache (e.g. embeddings.pkl)")
    args = parser.parse_args()

    # Device selection: MPS for Apple Silicon, CUDA for NVIDIA, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    image_paths = get_image_paths(args.dir)
    print(f"Found {len(image_paths)} images in {args.dir}")

    if not image_paths:
        print("No images found. Check your directory and supported formats.")
        return

    model, preprocess = load_model(device)
    data = embed_images(image_paths, model, preprocess, device, cache_path=args.cache)

    print(f'\nSearching for: "{args.query}"')
    results = search(args.query, data, model, device, top_k=args.top, threshold=args.threshold)

    print(f"\nFound {len(results)} matches\n")

    with open(args.out, "w") as f:
        for score, path in results:
            line = f"{score:.4f}\t{path}"
            print(line)
            f.write(line + "\n")

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()