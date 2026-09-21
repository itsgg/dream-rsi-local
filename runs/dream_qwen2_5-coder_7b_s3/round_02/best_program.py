import numpy as np
# MECHANISM: Use a logarithmic scale to prioritize bins with larger remaining capacity more strongly

def priority(item, bins):
    return -np.log(bins - item + 1)
