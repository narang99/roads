import json
from pathlib import Path
from mtrain.utils import json_to_content


def get_image_path(content) -> Path:
    ext_p = content["data"]["image"].split("?d=")[1]
    ext_p = f"/{ext_p}"
    return Path(ext_p)


def extract_single_id_json(iid, in_json_path_or_content, out_json_path):
    content = json_to_content(in_json_path_or_content)
    # if the content is list, assume the full file
    # else assume the small half file

    if isinstance(content, list):
        res = [c for c in content if c["id"] == iid]
    else:
        res = [in_json_path_or_content]
    if len(res) == 0:
        raise Exception(
            f"no result found for id={iid} in JSON={in_json_path_or_content}"
        )
    if len(res) == 1:
        with open(out_json_path, "w") as f:
            json.dump(res[0], f)
        return res[0]
    else:
        raise Exception(
            f"corrupted data: {in_json_path_or_content}, found multiple results for image_id={iid} data={res}"
        )
