from fastai.vision.all import SegmentationDataLoaders, get_image_files, unet_learner, Resize
import numpy as np
from fastai.callback.progress import CSVLogger, ProgressCallback


def get_learner(dls, model):
    learn = unet_learner(dls, model)
    learn.remove_cb(ProgressCallback)
    learn.add_cb(CSVLogger)
    return learn


def get_dls(bs, log_path, tile_size, images_root, masks_root) -> SegmentationDataLoaders:
    dls = SegmentationDataLoaders.from_label_func(
        log_path,
        bs=bs,
        fnames=get_image_files(images_root),
        label_func=lambda o: masks_root / f"{o.stem}.png",
        codes=np.array(["background", "trash"]),
        item_tfms=Resize(tile_size),
    )
    return dls