# given a label studio export, I want to prepare crops
# first go through all and create raw crops
# then provide the user a function for processing these
import cv2
import json
from typing import Iterator, Optional
from dataclasses import dataclass
from pathlib import Path
from mtrain.label_studio.crops import (
    extract_from_single_result,
    KMeansDatasetExplorer,
)
from mtrain.utils import mkdir, json_to_content



@dataclass
class SingleCrop:
    raw: Optional[Path]
    proc: Optional[Path]
    meta: Optional[Path]
    mask: Optional[Path]

    def parse_meta(self) -> Optional[dict]:
        if not self.meta.exists():
            return None
        with open(self.meta) as f:
            return json.load(f)


def _p_if_exists_else_none(p):
    if p.exists():
        return p
    else:
        None


class PrepareCrops:
    def __init__(self, out_dir: Path):
        self._o = out_dir
        mkdir(self._o)

    def get_kmeans_explorer(self, backdrop, idx=0):
        raws = [c.raw for c in self.get_all_existing_crops()]
        # the explorer maintains the directory structure correctly
        return KMeansDatasetExplorer(raws, backdrop, idx)

    def get_all_existing_crops_with_proc(self) -> Iterator[SingleCrop]:
        return filter(lambda crp: crp.proc is not None, self.get_all_existing_crops())

    def get_all_existing_crops(self) -> Iterator[SingleCrop]:
        for d in self._o.glob("*"):
            if d.is_dir():
                if not (d / "raw.png").exists():
                    continue
                yield SingleCrop(
                    raw=_p_if_exists_else_none(d / "raw.png"),
                    proc=_p_if_exists_else_none(d / "proc.png"),
                    meta=_p_if_exists_else_none(d / "meta.json"),
                    mask=_p_if_exists_else_none(d / "mask.json"),
                )

    def dump_raw_all(self, json_path_or_content):
        i = 0
        content = json_to_content(json_path_or_content)
        for c in content:
            fragments = extract_from_single_result(c)
            for _, frag in enumerate(fragments):
                self._dump_single_fragment(frag, self._o / str(i))
                i += 1

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
                        "path": frag.original.path,
                    },
                },
                f,
            )
