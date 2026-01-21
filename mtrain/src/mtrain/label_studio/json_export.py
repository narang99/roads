import json
from pathlib import Path

def get_image_path(content) -> Path:
    ext_p = content["data"]["image"].split("?d=")[1]
    ext_p = f"/{ext_p}"
    return Path(ext_p)



def extract_single_id_json(iid, in_json_path, out_json_path):
    with open(in_json_path) as f:
        content = json.load(f)
    res = [c for c in content if c["id"] == iid]
    if len(res) == 0:
        raise Exception(f"no result found for id={iid} in JSON={in_json_path}")
    if len(res) == 1:
        with open(out_json_path, "w") as f:
            json.dump(res[0], f)
        return res[0]
    else:
        raise Exception(f"corrupted file: {in_json_path}, found multiple results for image_id={iid} data={res}")
