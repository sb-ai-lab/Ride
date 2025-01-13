from abc import ABC, abstractmethod
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from multiprocessing import Pool
from typing import Callable

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from ride.clustering import AbstractCommunityResolver, Community
from ride.path_finding.dijkstra_pfa import AStar
from ride.path_finding.pfa import PathFinding

__all__ = [
    'HeuristicBuilder',
    'MinClusterDistanceBuilder'
]


class HeuristicBuilder(ABC):
    @abstractmethod
    def build_astar(self, g: nx.Graph, cms: AbstractCommunityResolver | Community) -> PathFinding:
        pass


@dataclass
class MinClusterDistanceCallable(Callable[[int, int], float]):
    nodes: list[dict]
    d_cluster: np.ndarray
    cluster: str

    def __call__(self, u: int, v: int) -> float:
        nodes = self.nodes
        d_clusters = self.d_cluster
        n1, n2 = nodes[u], nodes[v]
        c11 = n1[self.cluster]
        c12 = n2[self.cluster]
        return d_clusters[c11, c12]


@dataclass
class MinClusterDistanceBuilder(HeuristicBuilder):
    workers: int = 4
    cluster = 'cluster'

    def build_astar(self, g: nx.Graph, cms: AbstractCommunityResolver | Community) -> PathFinding:
        if isinstance(cms, AbstractCommunityResolver):
            cms = cms.resolve(g)
        w = self.workers
        cms_points = list(range(len(cms)))
        data = [(cms_points[i::w], self.cluster, cms, g) for i in range(w)]

        if self.workers == 1:
            d_clusters = calc(data[0])
        else:
            with Pool(w) as p:
                d_clusters = sum(tqdm(p.imap_unordered(calc, data), total=len(data)))

        nodes = g.nodes()
        return AStar(g, h=MinClusterDistanceCallable(nodes, d_clusters, cluster=self.cluster))


def dijkstra_pfa_min_dst(graph: nx.Graph,
                         start: set[int],
                         weight: str = 'length'
                         ) -> \
        dict[float]:
    adjacency = graph._adj
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    fringe = []
    for s in start:
        push(fringe, (0.0, next(c), s))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue
        dist[v] = d
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if u not in dist:
                push(fringe, (vu_dist, next(c), u))
    return dist


def calc(data):
    points, name, cms, g = data
    d_clusters = np.zeros((len(cms), len(cms)))

    for u in points:
        ll = dijkstra_pfa_min_dst(g, cms[u])
        q = {}
        for v, d in ll.items():
            if g.nodes()[v][name] in q:
                q[g.nodes()[v][name]] = min(q[g.nodes()[v][name]], d)
            else:
                q[g.nodes()[v][name]] = d
        for v in range(len(cms)):
            if v in q:
                d_clusters[u, v] = q[v]
    return d_clusters
