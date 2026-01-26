import hashlib
import sys
import json
import shutil
from functools import wraps
from pathlib import Path
from typing import Iterable, Callable





class SyntheticCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, fn_name: str, kwargs: dict, key_args: Iterable[str]):
        payload = {k: str(kwargs[k]) for k in key_args if k in kwargs}
        blob = json.dumps(payload, sort_keys=True).encode()
        h = hashlib.sha256(blob).hexdigest()[:16]
        return f"{fn_name}_{h}"

    def decorator(
        self,
        *,
        output_arg: str = "output_dir",
        num_samples_arg: str = "num_samples",
        key_args: Iterable[str],
        is_asset: Callable[[Path], bool] = lambda _: True
    ):
        """
        output_arg: name of output directory argument
        num_samples_arg: argument that specifies how many samples are requested
        key_args: arguments that define cache identity
        """

        def wrap(fn):
            @wraps(fn)
            def inner(*args, **kwargs):
                if output_arg not in kwargs:
                    raise ValueError(f"{output_arg} must be passed as kwarg")

                real_output_dir = Path(kwargs[output_arg])
                real_output_dir.mkdir(parents=True, exist_ok=True)

                num_samples = kwargs.get(num_samples_arg, None)

                cache_key = self._hash_key(fn.__name__, kwargs, key_args)
                cache_dir = self.root / cache_key
                cache_dir.mkdir(parents=True, exist_ok=True)


                print("CACHE: finding existing file count")
                cached_assets = filter(is_asset, cache_dir.rglob("*"))
                cached_count = sum(1 for _ in cached_assets)
                print("CACHE: existing count:", cached_count)

                # Enough samples → skip generation
                if num_samples is not None and cached_count >= num_samples:
                    print(f"samples already exist in cache, will copy to {cache_dir} -> {real_output_dir}")
                else:
                    # Otherwise generate into cache
                    kwargs[num_samples_arg] = num_samples - cached_count
                    kwargs[output_arg] = cache_dir
                    fn(*args, **kwargs)

                sys.stdout.flush()
                _copy_assets_only(cache_dir, real_output_dir, is_asset=is_asset, max_count=num_samples)
                sys.stdout.flush()



            return inner

        return wrap


# can replace copytree, keeping it here for later maybe
def _copy_assets_only(
    src: Path,
    dst: Path,
    *,
    is_asset,
    max_count: int,
):
    src = Path(src)
    dst = Path(dst)
    count = 0

    for path in src.rglob("*"):
        if not path.is_file():
            continue

        if not is_asset(path):
            continue

        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(path, out)
        count += 1
        if count >= max_count:
            break
    print(f"copied files: {count}")


def SuffixIn(suffs):
    suff_list = set(suffs)
    def wrapped(path: Path) -> bool:
        return path.suffix in suff_list
    return wrapped
            


DEFAULT_SYNTH_CACHE = SyntheticCache(Path.home() / ".mtrain_synth_cache")