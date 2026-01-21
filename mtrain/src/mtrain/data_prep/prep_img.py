# given a label studio export, we would like to prepare a directory
# containing all the images and some metadata
# the directory sturcture is a single directory per image
# each directory would have `image.png` and `export.json`
# export.json is the fragment of export found in label studio

from pathlib import Path
import shutil
from mtrain.label_studio.json_export import extract_single_id_json, get_image_path
from mtrain.utils import mkdir, json_to_content


class PrepareImages:
    def __init__(self, out_dir: Path):
        self._o = mkdir(out_dir)

    def add_raw_all(self, in_json_path_or_content):
        content = json_to_content(in_json_path_or_content)
        if not isinstance(content, list):
            raise Exception(
                f"the data passed to add_raw_all should be either a direct load of the full export file or the full export file path itself. Expected a list, got a dict. did you accidentally pass a single image's crop content? data={in_json_path_or_content}"
            )
        iids = [d["id"] for d in content]
        return [self.add_raw_single(iid, in_json_path_or_content) for iid in iids]

    def add_raw_single(self, iid: int, in_json_path_or_content) -> Path:
        # given the (main json file) json and iid, find and plce the image in out-dir
        # then place the extracted json in out-dir
        # you can als pass the simple json file
        in_json_path_or_content = json_to_content(in_json_path_or_content)
        out_dir = mkdir(self._o / str(iid))
        content = extract_single_id_json(
            iid, in_json_path_or_content, out_dir / "export.json"
        )
        src_ipath = get_image_path(content)
        shutil.copy(src_ipath, out_dir / f"raw{src_ipath.suffix}")
        return out_dir
