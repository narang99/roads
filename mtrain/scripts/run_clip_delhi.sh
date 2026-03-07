set -euxo pipefail
CF_NAME="$1"
QUERY="$2"

uv run clip_query.py \
	--dir /Users/hariomnarang/Desktop/personal/roads/mapillary_downloader/data/delhi/images \
	--cache $HOME/.roads-clip-cache/embeddings_delhi.pkl \
	--query "$QUERY" \
	--out "../../datasets/test-samples/neg-masking/V1/trash/clip_${CF_NAME}.txt"
