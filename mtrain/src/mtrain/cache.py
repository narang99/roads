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
        key_args: Iterable[str],
        is_asset: Callable[[Path], bool] = lambda _: True,
    ):
        """
        output_arg: name of output directory argument
        key_args: arguments that define cache identity
        """

        def wrap(fn):
            @wraps(fn)
            def inner(*args, **kwargs):
                if output_arg not in kwargs:
                    raise ValueError(f"{output_arg} must be passed as kwarg")

                real_output_dir = Path(kwargs[output_arg])
                if real_output_dir.exists():
                    print(f"output directory exists at {real_output_dir}, nuking")
                    shutil.rmtree(real_output_dir)
                real_output_dir.mkdir(parents=True, exist_ok=True)

                cache_key = self._hash_key(fn.__name__, kwargs, key_args)
                cache_dir = self.root / cache_key
                if cache_dir.exists():
                    print(f"CACHE HIT: will copy to {cache_dir} -> {real_output_dir}")
                else:
                    print("CACHE MISS: triggering data generation")
                    try:
                        # this sequence of actions have to be atomic
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        kwargs[output_arg] = cache_dir
                        fn(*args, **kwargs)
                    except Exception as ex:
                        # on failure, we mark that the cache does not exist
                        print(
                            f"failure in data generation function, deleting cache directory at {cache_dir}, reason={ex}. Will re-raise the exception, please do not interrupt, else the cache would be corrupted"
                        )
                        shutil.rmtree(cache_dir)
                        raise

                sys.stdout.flush()
                _copy_assets_only(
                    cache_dir, real_output_dir, is_asset=is_asset
                )
                sys.stdout.flush()

            return inner

        return wrap


# can replace copytree, keeping it here for later maybe
def _copy_assets_only(
    src: Path,
    dst: Path,
    *,
    is_asset,
):
    print(f"Copy: {src} {dst}")
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
    print(f"copied files: {count} to {dst}")


def SuffixIn(suffs):
    suff_list = set(suffs)

    def wrapped(path: Path) -> bool:
        return path.suffix in suff_list

    return wrapped


DEFAULT_SYNTH_CACHE = SyntheticCache(Path.home() / ".mtrain_synth_cache")
