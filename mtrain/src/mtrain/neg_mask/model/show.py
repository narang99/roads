from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fastai.vision.all import default_device
from torch.utils.data import DataLoader as TorchDataLoader

from .datasets.dataset import MaskClassificationDataset, _label_func
from mtrain.utils import show, overlay_mask_on_img


"""
usage examples

#cell 1
all_valid_preds, all_valid_targs, valid_decoded, all_valid_losses = get_preds_for_ds(learn, valid_ds)
all_train_preds, all_train_targs, train_decoded, all_train_losses = get_preds_for_ds(learn, train_ds)

#cell 2
train_losses_and_idxes = list(reversed(sorted((loss, i) for i, loss in enumerate(all_train_losses))))
valid_losses_and_idxes = list(reversed(sorted((loss, i) for i, loss in enumerate(all_valid_losses))))

#cell 3 — confusion matrix
show_confusion_matrix_using_preds(all_valid_preds, all_valid_targs, labels=LABELS)

#cell 4 — show worst loss examples
worst_idxs = [i for _, i in train_losses_and_idxes[:32]]
show_images(train_ds, worst_idxs, all_train_preds, labels=LABELS, title="Worst train losses")
"""

def get_preds_for_ds(learn, ds, bs=4, device=None):
    """Run inference over a dataset, return (preds, targs, decoded, losses)."""
    if device is None:
        device = default_device()

    loader = TorchDataLoader(ds, batch_size=bs, shuffle=False)
    learn.model.eval()
    learn.model.to(device)

    all_preds, all_targs, all_losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = learn.model(x)
            preds = logits.softmax(dim=1)
            losses = F.cross_entropy(logits, y.to(device), reduction="none")
            all_preds.append(preds.cpu())
            all_targs.append(y.cpu())
            all_losses.append(losses.cpu())

    all_preds = torch.cat(all_preds)
    all_targs = torch.cat(all_targs)
    all_losses = torch.cat(all_losses)
    decoded = all_preds.argmax(dim=1)
    return all_preds, all_targs, decoded, all_losses


def get_result_indices(targs, decoded, class_idx=None):
    """
    Return (correct_idxs, fp_idxs, fn_idxs).

    If class_idx is None:
        correct = model got it right
        fp = fn = all wrong predictions (class-agnostic)

    If class_idx is given (int):
        correct = model got it right for that class (targ == class_idx and decoded == class_idx)
        fp      = predicted class_idx but targ was something else
        fn      = targ was class_idx but model predicted something else
    """
    targs = targs.int()
    decoded = decoded.int()

    if class_idx is None:
        correct_idxs = (decoded == targs).nonzero().squeeze()
        wrong_idxs   = (decoded != targs).nonzero().squeeze()
        return correct_idxs, wrong_idxs, wrong_idxs

    c = class_idx
    correct_idxs = ((targs == c) & (decoded == c)).nonzero().squeeze()
    fp_idxs      = ((targs != c) & (decoded == c)).nonzero().squeeze()
    fn_idxs      = ((targs == c) & (decoded != c)).nonzero().squeeze()
    return correct_idxs, fp_idxs, fn_idxs


def show_classification_report(preds, targs, labels):
    from sklearn.metrics import classification_report
    decoded = preds.argmax(dim=1)
    print(classification_report(targs.int(), decoded.int(), target_names=labels))


def show_confusion_matrix(learn, dataloader, labels):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    preds, targs = learn.get_preds(dl=dataloader)
    pred_classes = preds.argmax(dim=1)
    cm = confusion_matrix(targs, pred_classes)
    fig, ax = plt.subplots(figsize=(30,30))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax)


def show_confusion_matrix_using_preds(preds, targs, labels):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    pred_classes = preds.argmax(dim=1)
    cm = confusion_matrix(targs, pred_classes)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()


def show_images(dataset, idxs, preds, labels, title, nrows=4):
    n = nrows * 4
    fig, axes = plt.subplots(nrows, 4, figsize=(30, 30))
    fig.suptitle(title)
    decoded = preds.argmax(dim=1)
    for ax, idx in zip(axes.flat, idxs[:n]):
        i = idx.item() if hasattr(idx, "item") else idx
        combined, targ = dataset[i]
        t_img = _get_image_from_tensor(combined)
        t_mask = combined[3].numpy().astype(bool)
        ax.imshow(t_img)
        _draw_mask_bboxes(ax, t_mask)
        pred_label = labels[decoded[i].item()]
        true_label = labels[int(targ.item())]
        conf = preds[i, decoded[i].item()].item()
        ax.set_title(f"{i} | pred: {pred_label} ({conf:.2f}) | true: {true_label}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_single(d, data_point):
    orig_img = plt.imread(str(d / "image.jpg"))
    orig_mask = plt.imread(str(d / "mask.png"))
    tf_img = denormalize(data_point[0][:3].permute(1, 2, 0))
    tf_mask = data_point[0][3]
    tf_img = (tf_img.numpy() * 255).astype(np.uint8)
    tf_mask = tf_mask.numpy().astype(bool)
    show(
        [
            orig_img,
            orig_mask,
            overlay_mask_on_img(orig_img, orig_mask.astype(bool)),
            tf_img,
            tf_mask,
            overlay_mask_on_img(tf_img, tf_mask),
        ],
        (10, 10),
        ncols=3,
        axis="off",
    )


def get_label_stats(dirs):
    counts = defaultdict(lambda: 0)
    for d in dirs:
        counts[_label_func(d)] += 1
    return {"counts": dict(counts)}


def denormalize(tf_img):
    std = torch.Tensor(MaskClassificationDataset.IMAGENET_STD)
    mean = torch.Tensor(MaskClassificationDataset.IMAGENET_MEAN)
    return (tf_img * std + mean).clamp(0, 1)


def _get_image_from_tensor(combined):
    return denormalize(combined[:3].permute(1, 2, 0))


def _draw_mask_bboxes(ax, mask):
    mask_uint8 = mask.astype(np.uint8)
    n, labeled = cv2.connectedComponents(mask_uint8)
    for i in range(1, n):
        ys, xs = np.where(labeled == i)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)