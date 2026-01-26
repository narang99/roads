from pathlib import Path
import random
import string
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


def compose(*funcs):
    def composed(x):
        for f in reversed(funcs):
            x = f(x)
        return x
    return composed


def pipe(*funcs):
    def piped(x):
        for f in funcs:
            x = f(x)
        return x
    return piped

def random_filename(k, suffix=None):
    # pathlib style suffix (inclues ., like ".jpg")
    chars = string.ascii_lowercase + string.digits
    name = "".join(random.choices(chars, k=k))
    if suffix is None:
        return name
    else:
        return f"{name}{suffix}"
