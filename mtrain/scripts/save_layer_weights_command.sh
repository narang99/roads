IMG_DIR_NAME="$1"

[ -n "$IMG_DIR_NAME" ]

echo uv run save_layer_weights.py \
"/Users/hariomnarang/Desktop/personal/roads/datasets/interpretation/smallnet/inputs/$IMG_DIR_NAME/image.jpg" \
 -o "/Users/hariomnarang/Desktop/personal/roads/datasets/interpretation/smallnet/analysis/$IMG_DIR_NAME" \
 --weights-dir /Users/hariomnarang/Desktop/personal/roads/datasets/interpretation/smallnet/weights
