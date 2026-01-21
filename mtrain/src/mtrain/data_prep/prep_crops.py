# given a label studio export, I want to prepare crops
# first go through all and create raw crops
# then provide the user a function for processing these
import cv2
import json
from typing import Iterator
from pathlib import Path
from mtrain.label_studio.crops import extract_from_single_result, KMeansDatasetExplorer



def _mkdir(p: Path):
    p.mkdir(exist_ok=True, parents=True)
    return p


def _json_to_dict(json_path_or_content) -> dict:
    content = json_path_or_content
    if isinstance(json_path_or_content, Path) or isinstance(json_path_or_content, str):
        with open(json_path_or_content) as f:
            content = json.load(f)
    return content


class PrepareCrops:
    def __init__(self, out_dir: Path):
        self._o = out_dir
        _mkdir(self._o)

    def get_kmeans_explorer(self):
        raws = list(self.get_all_raw_crops())
        # the explorer maintains the directory structure correctly
        return KMeansDatasetExplorer(raws)

    def get_all_raw_crops(self) -> Iterator[Path]:
        for d in self._o.glob("*"):
            if d.is_dir():
                cand = d / "raw.png"
                if cand.exists():
                    yield cand

    def add_raw_fragments(self, json_path_or_content):
        content = _json_to_dict(json_path_or_content)
        for c in content:
            self.add_raw_single(c)

    def add_raw_single(self, json_path_or_content):
        content = _json_to_dict(json_path_or_content)
        fragments = extract_from_single_result(content)
        for i, frag in enumerate(fragments):
            self._dump_single_fragment(frag, self._o / i)

    def _dump_single_fragment(self, frag, out_dir):
        out_dir = _mkdir(out_dir)
        cv2.imwrite(out_dir / "raw.png", frag.crop)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "bounding_box": frag.bounding_box,
                },
                f,
            )
