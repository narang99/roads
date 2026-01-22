import numpy as np

def direct_copy(background, frag, start0, start1):
    mask = np.any(frag != 0, axis=-1)
    end0 = start0 + mask.shape[0]
    end1 = start1 + mask.shape[1]
    roi = background[start0:end0, start1:end1]
    roi_row_len = roi.shape[0]
    roi_col_len = roi.shape[1]
    mask = mask[:roi_row_len, :roi_col_len]
    frag = frag[:roi_row_len, :roi_col_len]
    roi[mask] = frag[mask]
    return start0, start0 + roi_row_len, start1, start1 + roi_col_len

