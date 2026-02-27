from tqdm import tqdm
import argparse
import sys
from pathlib import Path
from mtrain.utils import mkdir



def make_symlinks(out):
    # for each image, create a folder
    # put image.jpg there
    # put the folder.jpg res there too
    dest_base = mkdir(out / "symlinks").resolve()
    for p in tqdm(list(out.rglob("res.jpg"))):
        p = p.resolve()
        if p.is_relative_to(dest_base):
            continue
        im_name = p.parent.name
        dest = mkdir(dest_base / im_name)
        if not (dest / "image.jpg").exists():
            link = dest / "image.jpg"
            link.symlink_to(p.parent / "image.jpg")
        res_dest_name = f"{p.parent.parent.name}.jpg"
        if not (dest / res_dest_name).exists():
            link = dest / res_dest_name
            link.symlink_to(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create symlinks for differential predictions organized by image."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output directory containing prediction results",
    )
    args = parser.parse_args()

    try:
        if not args.output_path.exists():
            raise FileNotFoundError(f"Output path does not exist: {args.output_path}")
        make_symlinks(args.output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
