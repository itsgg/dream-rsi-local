import numpy as np

# MECHANISM: Use a modified piecewise linear function with a focus on encouraging the use of bins with larger remaining capacities while avoiding worst-fit behavior.

def priority(item, bins):
    max_capacity = 150
    def s(b, item):
        g = b - item
        if g <= 2: return 4
        elif g <= 3: return 3
        elif g <= 5: return 2
        elif g <= 7: return 1
        elif g <= 9: return 0.9
        elif g <= 12: return 0.95
        elif g <= 15: return 0.97
        elif g <= 21: return 0.98
        else: return 0.99
    return np.array([s(b, item) for b in bins]) * (max_capacity - bins) / max_capacity
