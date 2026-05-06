import numpy as np
from typing import List, Any

def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average."""
    if not values or len(values) != len(weights):
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

def clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value between min and max."""
    return max(min_val, min(value, max_val))

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value between 0 and 1."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)
