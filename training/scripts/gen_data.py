import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path
import argparse

def numpy_to_yolo(start0, end0, start1, end1, img_height, img_width):
    """
    Convert numpy array slice coordinates to YOLO bounding box format.

    Args:
        start0, end0: row indices (y-axis)
        start1, end1: column indices (x-axis)
        img_height: total image height
        img_width: total image width

    Returns:
        x_center, y_center, width, height (all normalized to 0-1)
    """
    # Calculate box dimensions in pixels
    box_width = end1 - start1
    box_height = end0 - start0

    # Calculate center coordinates in pixels
    x_center_px = start1 + box_width / 2
    y_center_px = start0 + box_height / 2

    # Normalize to 0-1 range
    x_center = x_center_px / img_width
    y_center = y_center_px / img_height
    width = box_width / img_width
    height = box_height / img_height

    return x_center, y_center, width, height


def yolo_label(label_cls, start0, end0, start1, end1, img_height, img_width):
    x_center, y_center, width, height = numpy_to_yolo(
        start0, end0, start1, end1, img_height, img_width
    )
    return f"{label_cls} {x_center} {y_center} {width} {height}"


def merge_overlapping_boxes(boxes):
    """
    Merge overlapping bounding boxes.

    Args:
        boxes: list of tuples [(start0, end0, start1, end1), ...]

    Returns:
        merged_boxes: list of tuples with overlapping boxes merged
    """
    if not boxes:
        return []

    # Convert to numpy array for easier manipulation
    boxes = [list(box) for box in boxes]
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        # Start with current box
        current = boxes[i]
        used[i] = True

        # Keep merging until no more overlaps found
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                # Check if boxes overlap
                if boxes_overlap(current, boxes[j]):
                    # Merge boxes
                    current = [
                        min(current[0], boxes[j][0]),  # min start0
                        max(current[1], boxes[j][1]),  # max end0
                        min(current[2], boxes[j][2]),  # min start1
                        max(current[3], boxes[j][3]),  # max end1
                    ]
                    used[j] = True
                    changed = True

        merged.append(tuple(current))

    return merged


def boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.

    Args:
        box1, box2: [start0, end0, start1, end1]

    Returns:
        bool: True if boxes overlap
    """
    start0_1, end0_1, start1_1, end1_1 = box1
    start0_2, end0_2, start1_2, end1_2 = box2

    # Check for no overlap (easier to detect)
    if end0_1 <= start0_2 or end0_2 <= start0_1:  # No vertical overlap
        return False
    if end1_1 <= start1_2 or end1_2 <= start1_1:  # No horizontal overlap
        return False

    return True


def random_copy(background, frag, yolo_label_cls_id):
    max0, max1 = background[:, :, 0].shape
    start0, start1 = random.randint(0, max0), random.randint(0, max1)
    mask = np.any(frag != 0, axis=-1)
    end0 = start0 + mask.shape[0]
    end1 = start1 + mask.shape[1]
    roi = background[start0:end0, start1:end1]
    mask = mask[: roi.shape[0], : roi.shape[1]]
    frag = frag[: roi.shape[0], : roi.shape[1]]
    roi[mask] = frag[mask]
    return start0, end0, start1, end1


def random_multiset_total(objects, max_total=20):
    total = random.randint(0, max_total)
    result = []

    for _ in range(total):
        result.append(random.choice(objects))

    return result


def _get_image_splice_for_clustering_coords(image, flist):
    max_frag0 = max([frag.shape[0] for frag in flist])
    max_frag1 = max([frag.shape[1] for frag in flist])

    cluster_len0 = max_frag0 + random.randint(5, 100)
    cluster_len1 = max_frag1 + random.randint(5, 100)
    max0, max1 = image[:, :, 0].shape
    start0, start1 = random.randint(0, max0), random.randint(0, max1)
    end0 = start0 + cluster_len0
    end1 = start1 + cluster_len1
    return start0, end0, start1, end1
    # return image[start0:end0, start1:end1]


# def random_spimpose(image, frags, yolo_label_cls_id, total_count=30):
def random_spimpose(image, flist, yolo_label_cls_id):
    return [random_copy(image, frag, yolo_label_cls_id) for frag in flist]


def random_spimpose_with_cluster(
    image, frags, yolo_label_cls_id, max_clusters=3, total_count=30, max_per_cluster=20
):
    clusters = random.randint(0, max_clusters)
    flist = random_multiset_total(frags, total_count)
    # nothing to add
    if not flist:
        return ""
    boxes = []
    for _ in range(clusters):
        cluster_size = random.randint(0, max_per_cluster)
        random.shuffle(flist)
        frags_to_add = flist[:cluster_size]

        ss_s0, ss_e0, ss_s1, ss_e1 = _get_image_splice_for_clustering_coords(
            image, flist
        )
        subset_image = image[ss_s0:ss_e0, ss_s1:ss_e1]

        rel_box_coords = random_spimpose(subset_image, frags_to_add, yolo_label_cls_id)
        abs_box_coords = [
            (s0 + ss_s0, e0 + ss_s0, s1 + ss_s1, e1 + ss_s1)
            for (s0, e0, s1, e1) in rel_box_coords
        ]
        boxes.extend(abs_box_coords)
    boxes.extend(random_spimpose(image, flist, yolo_label_cls_id))
    boxes = merge_overlapping_boxes(boxes)
    labels = [
        yolo_label(
            yolo_label_cls_id,
            start0,
            end0,
            start1,
            end1,
            image.shape[0],
            image.shape[1],
        )
        for (start0, end0, start1, end1) in boxes
    ]
    return "\n".join(labels)


def draw_labels(img, label_content):
    img = img.copy()
    h, w = img.shape[0], img.shape[1]
    for line in label_content.split("\n"):
        class_id, x_c, y_c, width, height = map(float, line.strip().split())

        # Convert back to pixel coordinates
        x1 = int((x_c - width / 2) * w)
        y1 = int((y_c - height / 2) * h)
        x2 = int((x_c + width / 2) * w)
        y2 = int((y_c + height / 2) * h)

        # Draw rectangle
        color = plt.cm.tab10((int(class_id)) % 10)[:3]
        color = tuple(int(c * 255) for c in color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    plt.imshow(img)
    plt.show()


class RandomSampler:
    def __init__(self, images_base: Path, frags_base: Path, yolo_cls_label_id: int = 0):
        self.images = list(images_base.rglob("*.jpg"))
        self.frags = [cv2.imread(p) for p in frags_base.glob("*")]
        self._label_id = yolo_cls_label_id

    def __iter__(self):
        for p in self.images:
            img = cv2.imread(p)
            total_count = random.randint(0, 60)
            labels = random_spimpose_with_cluster(
                img, self.frags, self._label_id, total_count=total_count
            )
            yield img, labels

def _write_img_and_lbl(i, img, lbl, dest_folder):
    idest = dest_folder / "images" / f"{i}.jpg"
    ldest = dest_folder / "labels" / f"{i}.txt"
    cv2.imwrite(idest, img)
    with open(ldest, "w") as f:
        f.write(lbl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("-i", "--images", type=Path, required=True, help="Base directory for images")
    parser.add_argument("-f", "--fragments", type=Path, required=True, help="Base directory for fragments")
    parser.add_argument("-d", "--destination", type=Path, required=True, help="Destination folder for output")
    parser.add_argument("-n", "--num-samples", type=int, required=True, help="Number of samples to generate")

    args = parser.parse_args()

    image_base = args.images
    frags_base = args.fragments
    dest_folder = args.destination
    num_samples = args.num_samples

    dest_folder.mkdir(parents=True, exist_ok=True)
    (dest_folder / "images").mkdir(parents=True, exist_ok=True)
    (dest_folder / "labels").mkdir(parents=True, exist_ok=True)

    sampler = RandomSampler(image_base, frags_base)


    with tqdm(total=num_samples) as pbar:
        i = 0
        while i < num_samples:
            sampler = RandomSampler(image_base, frags_base)
            for (img, lbl) in iter(sampler):
                _write_img_and_lbl(i, img, lbl, dest_folder)
                i += 1
                print(i)
                pbar.update(i)
                if i >= num_samples:
                    break