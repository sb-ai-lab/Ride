import time
from typing import Any, Callable

import numpy as np


def get_optimal_clusters_number(nodes: int) -> int:
    alpha = 8.09 * (nodes ** (-0.48)) * (1 - 19.4 / (4.8 * np.log(nodes) + 8.8)) * nodes
    return int(alpha)

def compute(func: Callable, iterations=2, *args, **kwargs) -> tuple[float, Any]:
    result = None
    start = time.time()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end = time.time()
    return (end - start) / iterations, result
