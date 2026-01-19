# Data generation

- Find all fragments in label studio
- export the JSON (not min json)
- use the notebook  `extract_garbage` to extract fragments and put all of them in a flat directory
- `uv run scripts/gen_data.py -i "/Users/hariomnarang/Desktop/personal/roads/mapillary_downloader/data/samples" -d <out-dir> -n 10000 -f <frag-dir>` for generating data
- split the dataset
- compress
```bash
# Compress (fast, good default)
# gnu tar in mac is gtar
gtar -I "zstd -3" -cf archive.tar.zst dir/
- upload to s3

- Decompress in colab
```bash
# Decompress (normal)
tar --zstd -xf archive.tar.zst

# Decompress (parallel, fastest on multi-core)
tar -I "zstd -T0" -xf archive.tar.zst
```