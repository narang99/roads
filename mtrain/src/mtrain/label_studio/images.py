from pathlib import Path
import shutil
from mtrain.label_studio.json_export import extract_single_id_json, get_image_path


def prepare_image_folder(iid: int, in_json_path: Path, out_dir: Path) -> Path:
    # given the json and iid, find and plce the image in out-dir
    # then place the extracted json in out-dir
    out_dir = out_dir / str(iid)
    out_dir.mkdir(parents=True, exist_ok=True)
    content = extract_single_id_json(iid, in_json_path, out_dir / "export.json")
    src_ipath = get_image_path(content)
    shutil.copy(src_ipath, out_dir / f"image{src_ipath.suffix}")
    return out_dir
