from fastai.vision.all import SegmentationDataLoaders, get_image_files, unet_learner
import numpy as np
from fastai.callback.progress import CSVLogger, ProgressCallback


def get_learner(dls, model):
    learn = unet_learner(dls, model)
    learn.remove_cb(ProgressCallback)
    learn.add_cb(CSVLogger)
    return learn


def get_dls(bs, images_root, masks_root):
    dls = SegmentationDataLoaders.from_label_func(
        images_root,
        bs=bs,
        fnames=get_image_files(images_root),
        label_func=lambda o: masks_root / o.name,
        codes=np.array(["background", "trash"]),
    )
    return dls