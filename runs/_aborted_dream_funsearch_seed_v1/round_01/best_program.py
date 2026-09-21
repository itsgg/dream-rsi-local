import numpy as np

def priority(item, bins):
    penalty = 100 / (bins + item)
    penalty = np.where(penalty < 0, 0, penalty)
    return penalty
