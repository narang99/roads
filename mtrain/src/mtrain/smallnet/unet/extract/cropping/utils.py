import random

def get_annotation_box(coco, img_id, ann_idx):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    if ann_idx is not None:
        ann = anns[ann_idx]
    else:
        ann = random.choice(anns)
    x, y, w, h = ann["bbox"]
    return x, y, w, h