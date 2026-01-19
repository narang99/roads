import random
import numpy as np
from mtrain.yolo.tfms import mk_yolo_label_from_numpy_coords
from mtrain.synth.tfms import (
    merge_overlapping_boxes,
    random_blur,
    random_luminosity_scale,
)


def random_copy(background, frag):
    max0, max1 = background[:, :, 0].shape
    start0, start1 = random.randint(0, max0), random.randint(0, max1)
    mask = np.any(frag != 0, axis=-1)
    end0 = start0 + mask.shape[0]
    end1 = start1 + mask.shape[1]
    roi = background[start0:end0, start1:end1]
    roi_row_len = roi.shape[0]
    roi_col_len = roi.shape[1]
    mask = mask[:roi_row_len, :roi_col_len]
    frag = frag[:roi_row_len, :roi_col_len]
    roi[mask] = frag[mask]
    return start0, start0 + roi_row_len, start1, start1 + roi_col_len


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


def random_spimpose(image, flist):
    return [random_copy(image, frag) for frag in flist]


def random_spimpose_with_cluster(
    image,
    frags,
    yolo_label_cls_id,
    max_clusters=3,
    total_count=30,
    max_per_cluster=20,
    apply_tfms=False,
):
    clusters = random.randint(0, max_clusters)
    flist = random_multiset_total(frags, total_count)
    if not flist:
        return ""
    boxes = []
    for _ in range(clusters):
        cluster_size = random.randint(0, max_per_cluster)
        random.shuffle(flist)
        frags_to_add = flist[:cluster_size]
        if apply_tfms:
            frags_to_add = [random_luminosity_scale(f) for f in frags_to_add]

        ss_s0, ss_e0, ss_s1, ss_e1 = _get_image_splice_for_clustering_coords(
            image, flist
        )
        subset_image = image[ss_s0:ss_e0, ss_s1:ss_e1]

        rel_box_coords = random_spimpose(subset_image, frags_to_add)
        abs_box_coords = [
            (s0 + ss_s0, e0 + ss_s0, s1 + ss_s1, e1 + ss_s1)
            for (s0, e0, s1, e1) in rel_box_coords
        ]
        boxes.extend(abs_box_coords)
    boxes.extend(random_spimpose(image, flist))
    if apply_tfms:
        blurred = random_blur(image)
        for box in boxes:
            s0, e0, s1, e1 = box
            image[s0:e0, s1:e1] = blurred[s0:e0, s1:e1]
    boxes = merge_overlapping_boxes(boxes)
    labels = [
        mk_yolo_label_from_numpy_coords(
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

