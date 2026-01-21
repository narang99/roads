from pathlib import Path
import json


def mkdir(p: Path):
    p.mkdir(exist_ok=True, parents=True)
    return p


def json_to_content(json_path_or_content) -> dict:
    content = json_path_or_content
    if isinstance(json_path_or_content, Path) or isinstance(json_path_or_content, str):
        with open(json_path_or_content) as f:
            content = json.load(f)
    return content
