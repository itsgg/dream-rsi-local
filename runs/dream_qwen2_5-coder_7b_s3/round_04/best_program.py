import numpy as np
# MECHANISM: Use a piecewise linear function to prioritize bins based on remaining capacity.

def priority(item, bins):
    def s(b, item):
        g = b - item
        if g <= 10: return 1.0 - g / 10
        elif g <= 20: return 0.1
        else: return 0.05
    return np.array([s(b, item) for b in bins])
