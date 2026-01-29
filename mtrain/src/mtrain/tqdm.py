from tqdm import tqdm as orig_tqdm
from functools import partial

tqdm = partial(orig_tqdm, bar_format="{percentage:3.0f}%")

def progress(i, n, step=10):
    pct = int((i + 1) / n * 100)
    if pct % step == 0:
        print(f"{pct}%", flush=True)


class Progress:
    def __init__(self, n, title="", step=10):
        self.n = n
        self.step = step
        self.last_printed = [-1]
        self.title = title
        print(f"Progress track: {title} total_samples={self.n}", flush=True)
    
    def __call__(self, i):
        pct = int((i + 1) / self.n * 100)
        pct_step = (pct // self.step) * self.step
        if pct_step > self.last_printed[0]:
            print(self._get_pct_str(pct_step), flush=True)
            self.last_printed[0] = pct_step

    def _get_pct_str(self, pct_step):
        pct_str = f"{pct_step}%"
        if self.title:
            pct_str = f"{self.title}: {pct_str}"
        return pct_str



def make_progress(step=10):
    """
    Returns a progress function that prints only when progress crosses a new step.
    """
    last_printed = [-1]  # mutable so closure can update it

    def progress(i, n):
        pct = int((i + 1) / n * 100)
        pct_step = (pct // step) * step
        if pct_step > last_printed[0]:
            print(f"{pct_step}%", flush=True)
            last_printed[0] = pct_step

    return progress