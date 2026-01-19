from pathlib import Path
from tqdm import tqdm
import random
from mtrain.synth.v1 import random_spimpose_with_cluster
import cv2


class RandomSamplerV1:
    def __init__(self, images_base: Path, frags_base: Path, yolo_cls_label_id: int = 0):
        self.images = list(images_base.rglob("*.jpg"))
        self.frags = [cv2.imread(p) for p in frags_base.glob("*")]
        self._label_id = yolo_cls_label_id

    def __iter__(self):
        for p in self.images:
            yield self._apply_impose(p, False)
            max_copies = random.randint(1, 5)
            for _ in range(max_copies):
                yield self._apply_impose(p, True)

    def _apply_impose(self, p, apply_tfms):
        img = cv2.imread(p)
        total_count = random.randint(0, 60)
        labels = random_spimpose_with_cluster(
            img,
            self.frags,
            self._label_id,
            total_count=total_count,
            apply_tfms=apply_tfms,
        )
        return img, labels


def _write_img_and_lbl(i, img, lbl, dest_folder):
    idest = dest_folder / "images" / f"{i}.jpg"
    ldest = dest_folder / "labels" / f"{i}.txt"
    cv2.imwrite(idest, img)
    with open(ldest, "w") as f:
        f.write(lbl)


def generate_data(
    image_base: Path | str,
    frags_base: Path | str,
    dest_folder: Path | str,
    num_samples: int,
):
    image_base = Path(image_base)
    frags_base = Path(frags_base)
    dest_folder = Path(dest_folder)
    num_samples = Path(num_samples)

    dest_folder.mkdir(parents=True, exist_ok=True)
    (dest_folder / "images").mkdir(parents=True, exist_ok=True)
    (dest_folder / "labels").mkdir(parents=True, exist_ok=True)

    sampler = RandomSamplerV1(image_base, frags_base)

    with tqdm(total=num_samples) as pbar:
        i = 0
        while i < num_samples:
            sampler = RandomSamplerV1(image_base, frags_base)
            for img, lbl in iter(sampler):
                _write_img_and_lbl(i, img, lbl, dest_folder)
                i += 1
                pbar.update(i)
                if i >= num_samples:
                    break
