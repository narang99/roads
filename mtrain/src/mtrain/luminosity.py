from dataclasses import dataclass
import cv2
import numpy as np
import logging


def alter_using_lab(img_bgr, factor):
    """
    factor > 1.0 → brighter
    factor < 1.0 → darker
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= factor
    lab[..., 0] = np.clip(lab[..., 0], 0, 255)

    lab = lab.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@dataclass
class StdLumMatcherParams:
    src_mean: float
    src_std: float
    tgt_mean: float
    tgt_std: float

    def scale_target_mean(self, scale: float):
        self.tgt_mean *= scale


def get_std_matcher_params(src_bgr, tgt_bgr):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_L = src_lab[..., 0].reshape(-1)
    tgt_L = tgt_lab[..., 0].reshape(-1)

    src_mean, src_std = src_L.mean(), src_L.std()
    tgt_mean, tgt_std = tgt_L.mean(), tgt_L.std()
    return StdLumMatcherParams(
        src_mean=src_mean,
        src_std=src_std,
        tgt_mean=tgt_mean,
        tgt_std=tgt_std,
    )


def do_std_matching(
    img_bgr: np.ndarray, mask: np.ndarray, p: StdLumMatcherParams, eps: float = 1e-6
):
    print("params", p)
    # change luminosity of img_bgr at mask
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = mask.astype(bool)

    L = lab[..., 0]
    L_new = (L - p.src_mean) * (p.tgt_std / (p.src_std + eps)) + p.tgt_mean
    L[mask] = np.clip(L_new[mask], 0, 255)
    lab[..., 0] = L

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# def get_luminosity_matcher(src_bgr, tgt_bgr, eps=1e-6):
#     # Convert to LAB
#     src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
#     tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

#     # Extract L values
#     src_L = src_lab[..., 0]
#     tgt_L = tgt_lab[..., 0].reshape(-1)

#     # Stats
#     src_mean, src_std = src_L.mean(), src_L.std()
#     tgt_mean, tgt_std = tgt_L.mean(), tgt_L.std()

#     # Apply only on source object
#     L = src_lab[..., 0]
#     L_new = (L - src_mean) * (tgt_std / (src_std + eps)) + tgt_mean
#     L = np.clip(L_new, 0, 255)

#     src_lab[..., 0] = L
#     return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def match_luminosity(src_bgr, src_mask, tgt_bgr, tgt_mask, eps=1e-6):
    # matches luminosity of source to target
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mask = _mask_bool(src_mask)
    tgt_mask = _mask_bool(tgt_mask)

    # Extract masked L values
    src_L = src_lab[..., 0][src_mask]
    tgt_L = tgt_lab[..., 0][tgt_mask]

    if len(tgt_L) == 0:
        raise ValueError("Target mask is empty")

    # Stats
    src_mean, src_std = src_L.mean(), src_L.std()
    tgt_mean, tgt_std = tgt_L.mean(), tgt_L.std()


    # Apply only on source object
    # im not sure what this function is
    print("means src", src_mean, "tgt", tgt_mean)
    L = src_lab[..., 0]
    L_new = (L - src_mean) * (tgt_std / (src_std + eps)) + tgt_mean
    L[src_mask] = np.clip(L_new[src_mask], 0, 255)

    src_lab[..., 0] = L
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _mask_bool(mask):
    return mask.astype(bool)



def do_percentile_matching(
    src_low, src_high,
    tgt_low, tgt_high,
    img_bgr: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
):
    """
    Match luminosity using percentile statistics on LAB L channel.
    Applies only inside mask.
    """

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = mask.astype(bool)

    L = lab[..., 0]

    # scale + shift
    scale = (tgt_high - tgt_low) / (src_high - src_low + eps)
    shift = tgt_low - scale * src_low

    L_new = scale * L + shift
    L[mask] = np.clip(L_new[mask], 0, 255)

    lab[..., 0] = L
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def compute_percentiles(img_bgr, mask=None, p_low=50, p_high=95):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    vals = lab[..., 0]
    if mask:
        vals = vals[mask.astype(bool)]
    return (
        np.percentile(vals, p_low),
        np.percentile(vals, p_high)
    )


def do_gamma_luminosity(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    gamma: float,
):
    """
    Apply gamma correction on LAB L channel, only inside mask.
    gamma < 1 → brighter
    gamma > 1 → darker
    """

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = mask.astype(bool)

    L = lab[..., 0] / 255.0
    L_gamma = np.power(L, gamma) * 255.0

    L[mask] = np.clip(L_gamma[mask], 0, 255)

    lab[..., 0] = L * 255.0 / 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)