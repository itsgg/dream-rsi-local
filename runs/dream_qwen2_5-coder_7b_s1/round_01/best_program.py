import numpy as np
# MECHANISM: Use a linear decay function for preference scores based on remaining capacity.

def priority(item, bins):
    max_capacity = 150
    return max_capacity - bins
