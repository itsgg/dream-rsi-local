# MECHANISM: Use a combination of logarithmic function for remaining capacity and exponential decay for bin utilization to prioritize bins effectively

import numpy as np

def priority(item, bins):
    remaining_capacity = bins - item
    priority_scores = np.zeros_like(remaining_capacity, dtype=float)
    
    # Logarithmic component for remaining capacity
    log_component = -np.log(remaining_capacity + 1)  # Avoid log(0)
    
    # Exponential component for bin utilization
    utilization = bins / 150
    exp_component = np.exp(-utilization * 10)  # Decay factor
    
    # Combine both components
    priority_scores = log_component * exp_component
    
    # Apply a penalty for very empty bins
    penalty = np.maximum(0, bins - 10) / 45  # Penalty for bins with less than 10 capacity left
    priority_scores -= penalty
    
    return priority_scores
