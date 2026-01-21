# given a label studio export, I want to prepare crops
# first go through all and create raw crops
# then provide the user a function for processing these
import cv2
import json
from typing import Iterator
from pathlib import Path
from mtrain.label_studio.crops import extract_from_single_result, KMeansDatasetExplorer
from mtrain.utils import mkdir, json_to_content


class PrepareCrops:
    def __init__(self, out_dir: Path):
        self._o = out_dir
        mkdir(self._o)

    def get_kmeans_explorer(self):
        raws = list(self.get_all_existing_raw_crops())
        # the explorer maintains the directory structure correctly
        return KMeansDatasetExplorer(raws)

    def get_all_existing_raw_crops(self) -> Iterator[Path]:
        for d in self._o.glob("*"):
            if d.is_dir():
                cand = d / "raw.png"
                if cand.exists():
                    yield cand

    def dump_raw_all(self, json_path_or_content):
        content = json_to_content(json_path_or_content)
        for c in content:
            self.add_raw_single(c)

    def dump_raw_single(self, json_path_or_content):
        content = json_to_content(json_path_or_content)
        fragments = extract_from_single_result(content)
        for i, frag in enumerate(fragments):
            self._dump_single_fragment(frag, self._o / str(i))

    def _dump_single_fragment(self, frag, out_dir):
        out_dir = mkdir(out_dir)
        cv2.imwrite(out_dir / "raw.png", frag.crop)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "bounding_box": {
                        "r": frag.bounding_box.r,
                        "c": frag.bounding_box.c,
                        "r_len": frag.bounding_box.r_len,
                        "c_len": frag.bounding_box.c_len,
                    },
                    "original": {
                        "r_len": frag.original.r_len,
                        "c_len": frag.original.c_len,
                    }
                },
                f,
            )
