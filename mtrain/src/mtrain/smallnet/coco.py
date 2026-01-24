# given a coco file, some util functions
from pathlib import Path
from typing import Optional
import numpy as np
import json
from pycocotools.coco import COCO


class CocoAnnotationFile:
    def __init__(self, ann_file):
        with open(ann_file) as f:
            self._content = json.load(f)
        self._fname_by_ann = {
            _key(img["file_name"]): img
            for img in self._content["images"]
        }
        self._coco = COCO(ann_file)

    def extract_bboxes_with_path(self, path) -> Optional[list[np.ndarray]]:
        fname = _key(path)
        item = self._fname_by_ann.get(fname)
        if not item:
            return None
        return self.extract_bboxes_with_id(item["id"])

    def extract_bboxes_with_id(self, img_id) -> list[np.ndarray]:
        annIds = self._coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
        anns = self._coco.loadAnns(annIds)
        # [x, y, w, h]
        return [
            [int(a) for a in ann["bbox"]]
            for ann in anns
        ]

def _key(img_path):
    parts = Path(img_path).parts
    return f"{parts[-2]}/{parts[-1]}"


# vibe-coded
def merge_overlapping_bboxes(bboxes):
    """
    Merge bounding boxes that have any non-zero overlap.
    bbox: [x,y,w,h]
    bboxes: list[bbox]
    """
    if len(bboxes) == 0:
        return []
    
    # Convert to [x1, y1, x2, y2] format
    boxes = np.array([[x, y, x+w, y+h] for x, y, w, h in bboxes])
    print(boxes)
    
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
            
        current = boxes[i].copy()
        used[i] = True
        
        # Keep merging until no more overlaps found
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                
                # Check for any overlap
                x1 = max(current[0], boxes[j][0])
                y1 = max(current[1], boxes[j][1])
                x2 = min(current[2], boxes[j][2])
                y2 = min(current[3], boxes[j][3])
                
                has_overlap = (x2 > x1) and (y2 > y1)
                
                if has_overlap:
                    # Merge boxes
                    current[0] = min(current[0], boxes[j][0])
                    current[1] = min(current[1], boxes[j][1])
                    current[2] = max(current[2], boxes[j][2])
                    current[3] = max(current[3], boxes[j][3])
                    used[j] = True
                    changed = True
        
        # Convert back to [x, y, w, h]
        merged.append([current[0], current[1], current[2]-current[0], current[3]-current[1]])
    return merged