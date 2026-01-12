# Mapillary Downloader

Download all street-level images for a city from Mapillary.

## Installation

```bash
uv sync
```

## Usage

Set your Mapillary access token:

```bash
export MAPILLARY_ACCESS_TOKEN="your_token_here"
```

Run the downloader:

```bash
uv run python -m mapillary_downloader --city "Palo Alto, CA" --output ./images
```

## Options

- `--city`: City name to download images for (required)
- `--output`: Output directory for images (default: `./output`)
- `--size`: Image size: `thumb_256_url`, `thumb_1024_url`, `thumb_2048_url`, `thumb_original_url` (default: `thumb_1024_url`)
- `--state-db`: Path to SQLite state database (default: `<output>/state.db`)



# Labelling
- label-studio, i can also deploy on railway: https://railway.com/deploy/label-studio
