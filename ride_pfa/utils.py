import random
import time
from typing import Any, Callable, Optional

import networkx as nx
import numpy as np

__all__ = [
    'get_optimal_clusters_number',
    'get_execution_time',
    'add_inverse_edges_weight',
    'generate_points'
]


def get_optimal_clusters_number(nodes: int) -> int:
    alpha = 8.09 * (nodes ** (-0.48)) * (1 - 19.4 / (4.8 * np.log(nodes) + 8.8)) * nodes
    return int(alpha)


def get_execution_time(func: Callable, iterations=2, *args, **kwargs) -> tuple[float, Any]:
    result = None
    start = time.time()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end = time.time()
    return (end - start) / iterations, result


def add_inverse_edges_weight(
        g: nx.Graph,
        weight: str,
        inverse_weight: Optional[str] = None,
        eps: float = 0.0001) -> str:
    """

    Parameters
    ----------
    g - nx.Graph
    weight - name of edges weight for inverse
    inverse_weight - optional inverse weight name. If None then  "inverse_{weight}" will be used.
    eps - epsilon to avoid division by 0
    Returns - name of inverse edge weight.
    -------

    """
    if inverse_weight is None:
        inverse_weight = f'inverse_{weight}'
    for u, v, d in g.edges(data=True):
        d[inverse_weight] = 1 / (d[weight] + eps)
    return inverse_weight


def get_node_for_initial_graph(graph: nx.Graph):
    """

    Parameters
    ----------
    graph - graph

    Returns - two different point in graph
    -------

    """
    nodes = list(graph.nodes())
    f, t = random.choice(nodes), random.choice(nodes)
    while f == t:
        f, t = random.choice(nodes), random.choice(nodes)
    return f, t


def generate_points(graph: nx.Graph, num: int = 1000) -> list[tuple[int, int]]:
    """

    Parameters
    ----------
    graph - graph
    num - number of points

    Returns list of start and end points
    -------

    """
    return [get_node_for_initial_graph(graph) for _ in range(num)]
