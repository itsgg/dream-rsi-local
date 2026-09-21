# MECHANISM: Penalize bins based on the gap between remaining capacity and item size, favoring larger gaps.

import numpy as np

def priority(item, bins):
    gaps = bins - item
    valid_gaps = np.maximum(gaps, 0)
    return -valid_gaps / (item + 1e-10)  # Avoid division by zero and favor larger gaps
