from mtrain.example_dir.bulk import generate_smallnet_50x50
from mtrain.example_dir.iterdir import get_dirs
from mtrain.neg_mask.openai_clip import get_images_from_clip_file
from mtrain.example_dir import run_bulk_inference, create_dirs_for_images
from pathlib import Path

################################# input images parameters #################################################
############# users should change this ############################
DELHI_SAMPLES_TEST = Path('/Users/hariomnarang/Desktop/personal/roads/datasets/inference/test_set')

DIRECTORIES = list(get_dirs(DELHI_SAMPLES_TEST))
##########################################################################################################


def main():
    generate_smallnet_50x50(DIRECTORIES)
    print("Processing complete")


if __name__ == "__main__":
    main()
