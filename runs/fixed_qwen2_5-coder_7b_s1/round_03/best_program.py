import numpy as np

# MECHANISM: Use a piecewise linear function with different slopes for different remaining capacities.

def priority(item, bins):
    capacities = np.clip(bins - item, 0, 150)
    scores = np.zeros_like(bins)
    thresholds = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    slopes = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    
    for i in range(len(thresholds)):
        mask = capacities <= thresholds[i]
        scores[mask] = slopes[i] * (thresholds[i] - capacities[mask])
        if np.any(mask):
            break
    
    return scores
