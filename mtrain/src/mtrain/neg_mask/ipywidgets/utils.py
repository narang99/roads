import io

import numpy as np
from PIL import Image


def arr_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a numpy array (uint8 RGB or grayscale) as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="png")
    buf.seek(0)
    return buf.getvalue()
