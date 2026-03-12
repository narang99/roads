import shutil

import ipywidgets as widgets
from IPython.display import display

from mtrain.smallnet.unet.extract.draw import overlay_mask_on_img
from mtrain.neg_mask.model.datasets.dataset import (
    MaskClassificationDataset,
    denormalize,
)
from mtrain.neg_mask.ipywidgets.utils import arr_to_png_bytes


def get_preds_for_ds(learn, ds, device=None, bs=4):
    from torch.utils.data import DataLoader as TorchDataLoader
    import torch.nn.functional as F
    import torch
    from fastai.vision.all import default_device

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
    TRASH, OTHER = 0, 1
    fp_idxs = ((all_targs == OTHER) & (decoded == TRASH)).nonzero().squeeze()
    fn_idxs = ((all_targs == TRASH) & (decoded == OTHER)).nonzero().squeeze()
    return all_preds, all_targs, decoded, fp_idxs, fn_idxs, all_losses


_LABEL_BUTTON_STYLE: dict[str, str] = {
    "trash": "danger",
    "other": "info",
    "unknown": "warning",
}


class LossWidget:
    """
    Browse the highest-loss predictions from a MaskClassificationDataset,
    similar to fastai's ClassificationInterpretation.

    For each item shows:
      - crop image with mask overlay
      - true label, predicted label, per-class probabilities, loss
      - relabel buttons that move the sample directory on disk

    Usage
    -----
        w = LossWidget(learn, ds, n=50)
        w.ui()

    Parameters
    ----------
    learn : fastai Learner
    ds    : MaskClassificationDataset  (train=False)
    n     : number of top-loss items to browse  (default 50)
    theme : "Dark" | "Light"
    bs    : inference batch size
    """

    def __init__(
        self,
        learn,
        ds: MaskClassificationDataset,
        n: int = 50,
        theme: str = "Dark",
        bs: int = 4,
        dry_run: bool = True,
        descending: bool = False,
    ):
        self._ds = ds
        self._labels = ds.labels
        self._bg = {"Dark": "#1e1e1e", "Light": "#f5f5f5"}[theme]
        self._dry_run = dry_run

        print("Running inference…")
        preds, targs, decoded, _, _, losses = get_preds_for_ds(learn, ds, bs=bs)
        self._preds = preds  # [N, C]
        self._targs = targs  # [N]
        self._decoded = decoded  # [N]
        self._losses = losses  # [N]

        sorted_idxs = losses.argsort(descending=descending)
        self._sorted_idxs = sorted_idxs[:n].tolist()
        self._n = len(self._sorted_idxs)
        self._pos = 0

        # tracks ds_idx -> new label for items relabeled this session
        self._relabeled: dict[int, str] = {}
        print(f"Ready — {self._n} items to browse.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ui(self):
        """Render the widget. Call once per notebook cell."""
        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._out_img = widgets.Output()
        self._out_img.layout.background = self._bg

        self._lbl_status = widgets.Label()
        self._lbl_info = widgets.HTML()
        self._lbl_action = widgets.HTML(layout=widgets.Layout(margin="4px 0 0 0"))

        self._btn_prev = widgets.Button(
            description="◀ Prev",
            layout=widgets.Layout(width="90px"),
        )
        self._btn_next = widgets.Button(
            description="Next ▶",
            layout=widgets.Layout(width="90px"),
        )
        self._btn_prev.on_click(lambda _: self._go(-1))
        self._btn_next.on_click(lambda _: self._go(1))

        self._overlay_toggle = widgets.ToggleButton(
            value=True,
            description="Overlay on",
            icon="eye",
            layout=widgets.Layout(width="120px"),
        )
        self._overlay_toggle.observe(lambda _: self._render(), names="value")

        self._label_btns: dict[str, widgets.Button] = {}
        for label in self._labels:
            btn = widgets.Button(
                description=f"→ {label}",
                button_style=_LABEL_BUTTON_STYLE.get(label, ""),
                layout=widgets.Layout(width="120px"),
            )
            btn.on_click(lambda _, lbl=label: self._relabel(lbl))
            self._label_btns[label] = btn

        self._btn_delete = widgets.Button(
            description="Delete",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="120px"),
        )
        self._btn_delete.on_click(lambda _: self._delete())

        nav_row = widgets.HBox(
            [self._btn_prev, self._btn_next, self._overlay_toggle],
            layout=widgets.Layout(gap="8px"),
        )
        label_row = widgets.HBox(
            [*self._label_btns.values(), self._btn_delete],
            layout=widgets.Layout(gap="8px", margin="8px 0 0 0"),
        )

        display(
            widgets.VBox(
                [
                    self._lbl_status,
                    self._lbl_info,
                    nav_row,
                    label_row,
                    self._lbl_action,
                    self._out_img,
                ]
            )
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        if self._n == 0:
            self._lbl_status.value = "No items."
            return

        pos = self._pos
        ds_idx = self._sorted_idxs[pos]
        d = self._ds.dirs[ds_idx]

        pred_label_idx = int(self._decoded[ds_idx].item())
        pred_label = self._labels[pred_label_idx]
        loss = self._losses[ds_idx].item()
        conf = self._preds[ds_idx][pred_label_idx].item()
        current_label = d.parent.name  # reflects any in-session relabeling

        relabeled_note = " ✓ relabeled" if ds_idx in self._relabeled else ""
        self._lbl_status.value = f"Top loss  {pos + 1} / {self._n}{relabeled_note}"

        correct = "✓" if pred_label == current_label else "✗"
        prob_strs = " &nbsp; ".join(
            f"<i>p({lbl})={self._preds[ds_idx][i].item():.1%}</i>"
            for i, lbl in enumerate(self._labels)
        )
        self._lbl_info.value = (
            f"<b>True:</b> <code>{current_label}</code> &nbsp;"
            f"<b>Pred:</b> <code>{pred_label}</code> {correct} &nbsp;"
            f"<b>Conf:</b> {conf:.1%} &nbsp;"
            f"<b>Loss:</b> {loss:.4f}<br>"
            f"{prob_strs}<br>"
            f"<small style='color:#888'>{d}</small>"
        )

        combined, _ = self._ds[ds_idx]
        img_np, mask_np = denormalize(combined)
        print("mask shape", mask_np.shape)
        alpha = 0.4 if self._overlay_toggle.value else 0.0
        overlaid = overlay_mask_on_img(img_np, mask_np.astype(bool), alpha=alpha)

        self._out_img.clear_output(wait=True)
        with self._out_img:
            display(
                widgets.Image(
                    value=arr_to_png_bytes(overlaid),
                    format="png",
                    layout=widgets.Layout(width="400px"),
                )
            )

        self._btn_prev.disabled = pos == 0
        self._btn_next.disabled = pos == self._n - 1

        for label, btn in self._label_btns.items():
            if label == current_label:
                btn.button_style = "success"
                btn.icon = "check"
            else:
                btn.button_style = _LABEL_BUTTON_STYLE.get(label, "")
                btn.icon = ""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _go(self, delta: int):
        self._pos = max(0, min(self._n - 1, self._pos + delta))
        self._lbl_action.value = ""
        self._render()

    def _delete(self):
        ds_idx = self._sorted_idxs[self._pos]
        d = self._ds.dirs[ds_idx]
        new_dir = d.parent.parent / "unknown" / d.name

        if self._dry_run:
            self._lbl_action.value = (
                f"<code style='color:#f90'>[dry run]</code> "
                f"<code>{d}</code><br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;→ <code>{new_dir}</code> &amp; removed from dataset"
            )
        else:
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(new_dir))
            self._ds.dirs.pop(ds_idx)

        self._render()

    def _relabel(self, new_label: str):
        ds_idx = self._sorted_idxs[self._pos]
        d = self._ds.dirs[ds_idx]
        if new_label == d.parent.name:
            self._lbl_action.value = (
                "<code style='color:#f90'>[dry run]</code> "
                "&nbsp;&nbsp;&nbsp;&nbsp;→ <code>Already same label</code>"
            )
            self._render()
            return  # already this label

        new_dir = d.parent.parent / new_label / d.name

        if self._dry_run:
            self._lbl_action.value = (
                f"<code style='color:#f90'>[dry run]</code> "
                f"<code>{d}</code><br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;→ <code>{new_dir}</code>"
            )
        else:
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(new_dir))
            self._ds.dirs[ds_idx] = new_dir
            self._relabeled[ds_idx] = new_label

        self._render()
