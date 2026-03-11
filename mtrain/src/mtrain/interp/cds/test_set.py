from pathlib import Path

OTHER_PATHS = [
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/1443199652726345_23.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/1536815244038325_0.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/43e3a9c4-7462-4cf4-ac4b-c7bd93d5e584_29.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/1132484334736347_25.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/2643521872502871_6.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/214319536787147_15.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/463661068073161_30.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/214319536787147_20.jpg"
    ),
]

TRASH_PATHS = [
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/2610561689091377_12.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/868693752231900_20.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1132072153938604_118.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/d3e42811-0d2e-485a-88fb-79bbfcef5214_66.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1225392251226169_47.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1132072153938604_96.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1223142505005166_21.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/2964791813765696_5.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/932277432170462_36.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1017733166824028_14.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/27bbe71a-6c51-4a26-acc5-f5a1cd7429b2_29.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/904411971827702_10.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1197058985243034_70.jpg"
    ),
    Path(
        "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/826269927077418_32.jpg"
    ),
]


def get_test_images():
    path_by_tag = {}
    for p in OTHER_PATHS:
        path_by_tag[p] = "other"
    for p in TRASH_PATHS:
        path_by_tag[p] = "trash"
    return path_by_tag


def get_path_by_orig_predictions():
    return {
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/1443199652726345_23.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/1536815244038325_0.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/43e3a9c4-7462-4cf4-ac4b-c7bd93d5e584_29.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/1132484334736347_25.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/2643521872502871_6.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/214319536787147_15.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/463661068073161_30.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/other/214319536787147_20.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/2610561689091377_12.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/868693752231900_20.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1132072153938604_118.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/d3e42811-0d2e-485a-88fb-79bbfcef5214_66.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1225392251226169_47.jpg"
        ): "other",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1132072153938604_96.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1223142505005166_21.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/2964791813765696_5.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/932277432170462_36.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1017733166824028_14.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/27bbe71a-6c51-4a26-acc5-f5a1cd7429b2_29.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/904411971827702_10.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/1197058985243034_70.jpg"
        ): "trash",
        Path(
            "/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/neg-masking/V1/rocks/classification/simple_crop_level/trash/826269927077418_32.jpg"
        ): "trash",
    }
