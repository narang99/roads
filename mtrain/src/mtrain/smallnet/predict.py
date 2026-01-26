import cv2
from torchvision import transforms
from fastai.vision.all import PILImage
import numpy as np
import torch
from fastai.vision.all import PILImage
from typing import List, Tuple
from tqdm import tqdm
from mtrain.seg.cityscapes import cached_predict, CityScapesCls, get_mask_with_labels


def segment_tile_and_predict(image_path, learner, tile_size, add_labels=True):
    pred = cached_predict(image_path)
    mask = get_mask_with_labels(
        pred, [CityScapesCls.ROAD, CityScapesCls.SIDEWALK, CityScapesCls.TERRAIN]
    )
    # return tile_image_and_predict_vec_my(image_path, learner, tile_size, add_labels, mask)
    return tile_image_and_predict(image_path, learner, tile_size, add_labels, mask)

def tile_image_and_predict_vec_my(image_path, learner, tile_size, add_labels=True, mask=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tiles, coords = split_image_into_tiles_vec(img, tile_size)
    batch_results = batch_inference_mobilenet(tiles, learner)

    result = []

    tiles_send = []
    for t, coord, res in zip(tiles, coords, batch_results):
        # print(coord)
        r, c = coord
        if mask is not None and not mask[r, c]:
            # not in mask, always false
            result.append(False)
        else:
            result.append(res == "pos")
        tiles_send.append((t, (r,c)))

    # for tile in tqdm(tiles):
    #     t, (r, c) = tile
    #     if mask is not None and not mask[r, c]:
    #         # not in mask, always false
    #         result.append(False)
    #     else:
    #         if learner.predict(t)[0] == "pos":
    #             result.append(True)
    #         else:
    #             result.append(False)

    return highlight_tiles_bbox_with_index(
        img, tiles_send, result, tile_size, add_labels=add_labels
    )


def tile_image_and_predict(image_path, learner, tile_size, add_labels=True, mask=None, tfms=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tiles = split_image_into_tiles(img, tile_size)

    result = []

    for tile in tqdm(tiles):
        t, (r, c) = tile
        if mask is not None and not mask[r, c]:
            # not in mask, always false
            result.append(False)
        else:
            if tfms is not None:
                t = tfms(t)
            if learner.predict(t)[0] == "pos":
                result.append(True)
            else:
                result.append(False)

    return highlight_tiles_bbox_with_index(
        img, tiles, result, tile_size, add_labels=add_labels
    )


def split_image_into_tiles(
    img: np.ndarray, tile_size: int = 50
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Split an image into tiles of size tile_size x tile_size.
    Edge tiles may be smaller.

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W) or (H, W, C)
    tile_size : int
        Size of each tile (default 50)

    Returns
    -------
    tiles : list of (tile, (y, x))
        tile: cropped image patch
        (y, x): top-left coordinate of the tile in original image
    """
    H, W = img.shape[:2]
    tiles = []

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = img[y : y + tile_size, x : x + tile_size]
            tiles.append((tile, (y, x)))

    return tiles


def highlight_tiles_bbox_with_index(
    img: np.ndarray,
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    positives: List[bool],
    tile_size: int = 50,
    color=(0, 255, 0),
    thickness=2,
    font_scale=0.5,
    text_color=(255, 255, 255),
    add_labels=True,
) -> np.ndarray:
    """
    Draw bounding boxes and tile index for positive tiles.
    """
    img_out = img.copy()
    H, W = img.shape[:2]

    for idx, ((_, (y, x)), is_pos) in enumerate(zip(tiles, positives)):
        if not is_pos:
            continue

        y2 = min(y + tile_size, H)
        x2 = min(x + tile_size, W)

        # Draw rectangle
        cv2.rectangle(img_out, (x, y), (x2, y2), color, thickness)

        if add_labels:
            # Draw tile index
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Background for text (better visibility)
            cv2.rectangle(img_out, (x, y), (x + tw + 4, y + th + 4), color, -1)

            cv2.putText(
                img_out,
                label,
                (x + 2, y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

    return img_out

def tile_image_and_predict_vec(
    image_path,
    learner,
    tile_size,
    add_labels=True,
    mask=None,
):
    # --- load image ---
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- tile ---
    tiles, coords = split_image_into_tiles_vec(img, tile_size)

    # --- batched prediction ---
    # fastai predict returns (labels, idxs, probs)
    labels = batch_inference_mobilenet(tiles, learner)
    # labels = batch_predict(learner, tiles)  # returns list of labels
    positives = np.array(labels) == "pos"
    # labels, _, _ = learner.predict(tiles)
    # positives = (np.array(labels) == "pos")

    # --- vectorized mask handling ---
    if mask is not None:
        ys, xs = coords[:, 0], coords[:, 1]
        positives &= mask[ys, xs]

    # --- draw results ---
    tiles_with_coords = list(zip(tiles, map(tuple, coords)))

    return highlight_tiles_bbox_with_index(
        img,
        tiles_with_coords,
        positives.tolist(),
        tile_size,
        add_labels=add_labels,
    )


def split_image_into_tiles_vec(
    img: np.ndarray, tile_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into tiles and return a numpy array batch of shape
    (N, tile_size, tile_size, C), padding edge tiles as needed.
    
    Returns:
        tiles  : (N, tile_size, tile_size, 3) array
        coords : (N, 2) array of top-left (y, x) coords
    """
    H, W = img.shape[:2]

    ys = np.arange(0, H, tile_size)
    xs = np.arange(0, W, tile_size)

    coords = np.array([(y, x) for y in ys for x in xs], dtype=np.int32)
    N = len(coords)

    # Create a batch of zeros to hold tiles
    tiles_batch = np.zeros((N, tile_size, tile_size, 3), dtype=img.dtype)

    for i, (y, x) in enumerate(coords):
        tile = img[y:y + tile_size, x:x + tile_size]
        h, w = tile.shape[:2]
        tiles_batch[i, :h, :w] = tile  # pad automatically

    return tiles_batch, coords


def tiles_to_tensor(learn, tiles, device=None):
    """
    Convert a list/array of tiles to a fastai batch tensor.
    tiles: list of HxWx3 uint8 images or (N,H,W,3) array
    """
    # Convert to PILImage for fastai transforms
    pil_tiles = [PILImage.create(tile) for tile in tiles]
    
    # Apply learner's transforms
    batch = torch.stack([learn.dls.after_item(p) for p in pil_tiles])  # after_item is transforms
    # batch = batch.to(learn.model.device if device is None else device)
    
    return batch

# def batch_predict(learn, tiles):
#     batch = tiles_to_tensor(learn, tiles)
#     learn.model.eval()
#     with torch.no_grad():
#         logits = learn.model(batch)
    
#     # Get predicted class indices
#     preds = logits.argmax(dim=1)
    
#     # Map indices back to labels
#     labels = [learn.dls.vocab[i] for i in preds.cpu().numpy()]
    
#     return labels

# def predict_batch(learner, item, rm_type_tfms=None, with_input=False):
#     dl = learner.dls.test_dl(item, tfms=rm_type_tfms, num_workers=0)
#     inp,preds,_,dec_preds = learner.get_preds(dl=dl, with_input=True, with_decoded=True)
#     print(inp, preds, dec_preds)
#     i = getattr(learner.dls, 'n_inp', -1)
#     inp = (inp,) if i==1 else tuplify(inp)
#     dec_inp, nm = zip(*learner.dls.decode_batch(inp + tuplify(dec_preds)))
#     res = preds,nm,dec_preds
#     if with_input: res = (dec_inp,) + res
#     return res


def batch_inference_mobilenet(tiles, learner, device=None):
    """
    Run batch inference on a list or array of HxWxC tiles using MobileNet.
    
    Args:
        tiles  : list of HxWx3 uint8 numpy arrays (from cv2)
        model  : PyTorch MobileNet model (pretrained or your learner.model)
        device : 'cuda' or 'cpu' (default: GPU if available)
    
    Returns:
        preds : list of predicted class labels
    """
    model = learner
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # --- convert tiles to PIL and then to tensor ---
    pil_tiles = [PILImage.create(tile[..., ::-1]) for tile in tiles]  # BGR->RGB
    transform = transforms.Compose([
        transforms.ToTensor(),  # scales 0-255 -> 0-1 and HWC->CHW
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    batch_tensor = torch.stack([transform(p) for p in pil_tiles]).to(device)  # shape (N,3,H,W)
    
    # --- forward pass ---
    with torch.no_grad():
        logits = model(batch_tensor)            # (N, num_classes)
        preds_idx = logits.argmax(dim=1).cpu() # predicted class indices
    
    # --- map indices to labels if model has classes (fastai learner) ---
    vocab = learner.dls.vocab
    preds = [vocab[i] for i in preds_idx]
    return preds