from fastai.vision.all import load_learner
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from mtrain.smallnet.unet.v2_predict import predict_unet
from mtrain.tqdm import Progress
from pathlib import Path

IMAGES = [
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/133855545702381.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/229528722273731.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/236879098235008.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/252987033288051.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/269626184842101.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/272196584589826.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/307523204081522.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/345173251842687.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1060415194365648.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1087879079930186.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1102231600254227.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1163966157381398.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1198225783960786.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1199990993777444.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1200581373711298.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1305290797783116.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1451445016080306.jpg',
    '/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/positive-samples/1547254768966807.jpg',
]

# 50x50
MODEL_PATH = "/Users/hariomnarang/Desktop/personal/roads/datasets/models/iter_4_engulf_t009_more-skew-resnet18-v2.pkl"
SIZE = 50
DEST_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/small_positive_samples")

# 200x200
# MODEL_PATH = "/Users/hariomnarang/Desktop/gdrive-sync/garbage/experiments/T009-engulf-3000-200/log/export_iter_4.pkl"
# SIZE = 200
# DEST_DIR = Path("/Users/hariomnarang/Desktop/personal/roads/datasets/test-samples/200_small_positive_samples")

def main():
    learner = load_learner(MODEL_PATH)
    progress = Progress(len(IMAGES))
    for i, img in enumerate(IMAGES):
        img = Path(img)
        res, mask = predict_unet(img, SIZE, learner, strides=list(range(10, 191, 10)))
        dest = DEST_DIR / img.stem
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(res).convert("RGB").save(dest / "res.jpeg")
        Image.fromarray(mask, "L").save(dest / "mask.png")
        shutil.copy(img, dest / f"image{img.suffix}")
        progress(i)

if __name__ == '__main__':
    main()