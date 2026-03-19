from mtrain.neg_mask.crops import get_region_crops, padded_crop, bbox_only_mask
from mtrain.disk import DiskBooleanMask, DiskImage

class MissedExampleWidget:
    def __init__(self, dirs, pad=224):
        self.dirs = dirs
        self.pad = pad

    def get_crops_for_single_dir(self, index):
        miss_mask_path = self.dirs[index] / "negmask-miss.png"
        image_path = self.dirs[index] / "image.jpg"
        if not miss_mask_path.exists():
            return None
        mask = DiskBooleanMask.load(miss_mask_path)
        image = DiskImage.load(image_path)
        bboxes = list(get_region_crops(mask))

        res = []
        for bbox in bboxes:
            crop, _, _ = padded_crop(image, bbox, self.pad)
            crop_mask = bbox_only_mask(mask, bbox, self.pad)
            res.append((crop, crop_mask))
        return res