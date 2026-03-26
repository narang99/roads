# copy ds 1 images to main ds 2
import shutil
from mtrain.utils import mkdir, globL
from pathlib import Path
from tqdm import tqdm

def copy_negmask_ds_to_ds(src_ds, dest_ds, suffix=None):
    src_ds, dest_ds = Path(src_ds), Path(dest_ds)
    mkdir(dest_ds / "train")
    mkdir(dest_ds / "masks")

    for im in tqdm(globL(src_ds / "train", "*.jpg")):
        mask = src_ds / "masks" / f'{im.stem}.png'
        if not mask.exists():
            continue

        name = im.stem
        if suffix is not None:
            name = f"{name}_{suffix}"
        dest_im = dest_ds / "train" / f"{name}.jpg"
        dest_mask = dest_ds / "masks" / f"{name}.png"

        # print("COPY:", im, "->", dest_im)
        shutil.copy(im, dest_im)
        # print("COPY:", mask, "->", dest_mask)
        shutil.copy(mask, dest_mask)

