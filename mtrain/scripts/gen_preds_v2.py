from mtrain.example_dir.iterdir import get_dirs
from mtrain.neg_mask.openai_clip import get_images_from_clip_file
from mtrain.example_dir import run_bulk_inference, create_dirs_for_images
from pathlib import Path

################################# input images parameters #################################################
############# users should change this ############################
DELHI_SAMPLES_TEST = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/inference/delhi_sample_500_results_test")
DIRECTORIES = list(get_dirs(DELHI_SAMPLES_TEST))
# CLIP_NAME = "delhi_litter"
# CLIP_FILE = f"/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/trash/clip_{CLIP_NAME}.txt"
# CLS_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/crop_level")

# TOTAL_SAMPLES = 100
# DEST_DIR = Path(
#     "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/full_image_runs/delhi_litter_v2"
# )

# def get_images_not_in_train_set():
#     images = get_images_from_clip_file(CLIP_FILE)
#     all_dirs = set()
#     for label in ["other", "trash"]:
#         sample_names_without_crop_idx = (p.name.split("_")[0] for p in (CLS_DIR / label).glob("*") if p.is_dir())
#         for s in sample_names_without_crop_idx:
#             all_dirs.add(s)
#     return [img for img in images if img.stem not in all_dirs]

# IMAGES = list(get_images_not_in_train_set())[:TOTAL_SAMPLES]
##########################################################################################################


def main():
    # directories = create_dirs_for_images(IMAGES, DEST_DIR)
    edirs = run_bulk_inference(DIRECTORIES)
    for edir in edirs:
        print(edir.d)
    print("Processing complete")


if __name__ == "__main__":
    main()