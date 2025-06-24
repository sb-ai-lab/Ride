import logging as log
from abc import ABC, abstractmethod
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count as count
from typing import NewType, Union, Optional

import networkx as nx
from tqdm.auto import trange

__all__ = [
    'Community',
    'AbstractCommunityResolver',
    'LouvainCommunityResolver',
    'LouvainKMeansCommunityResolver'
]

Community = NewType('Community', Union[list[set[int]], tuple[set[int]]])


@dataclass
class AbstractCommunityResolver(ABC):
    weight: str = 'length'
    cluster: str = 'cluster'
    write_community: bool = True

    def resolve(self, g: nx.Graph) -> Community:
        cms: Community = self.do_resolve(g=g)
        if self.write_community:
            return validate_cms(g, cms, cluster_name=self.cluster)
        else:
            return cms

    @abstractmethod
    def do_resolve(self, g: nx.Graph) -> Community:
        pass


@dataclass
class LouvainCommunityResolver(AbstractCommunityResolver):
    resolution: float = 1
    seed: float = 1534

    def do_resolve(self, g: nx.Graph) -> Community:
        communities = nx.community.louvain_communities(g,
                                                       seed=self.seed,
                                                       weight=self.weight,
                                                       resolution=self.resolution)
        return communities


@dataclass
class LouvainKMeansCommunityResolver(LouvainCommunityResolver):
    """
        A graph clustering algorithm that takes Louvain's algorithm (for finding centroids) as a basis and then applies K-means
    """
    """
        Number of Iteration for K-means
    """

    max_iteration: int = 20
    """
        Print log and progress bar
    """
    print_log: bool = False
    """
        The weight that will be used in K-Means to find the centroid and cluster membership of a point
    """
    k_means_weight: Optional[str] = None

    def do_resolve(self, g: nx.Graph) -> Community:
        communities = super().resolve(g)
        return self.do_resolve_kmeans(g, communities)

    def do_resolve_kmeans(self, g: nx.Graph, communities: Community) -> Community:
        if self.print_log:
            log.info(f'communities: {len(communities)}')
        k_means_weight = self.k_means_weight if self.k_means_weight else self.weight
        _iter = trange(self.max_iteration) if self.print_log else range(self.max_iteration)
        do = True
        for _ in _iter:
            if not do:
                continue
            centers = []
            for i, cls in enumerate(communities):
                gc = g.subgraph(communities[i])
                center = nx.barycenter(gc, weight=k_means_weight)[0]
                centers.append(center)

            node2cls = k_means(g, centers, weight=k_means_weight)
            do = False
            for u, i in node2cls.items():
                if u not in communities[i]:
                    do = True
                    break
            if not do:
                continue

            communities = [set() for _ in range(len(centers))]
            for u, c in node2cls.items():
                communities[c].add(u)
            communities = validate_cms(g, communities, cluster_name=self.cluster)
        return communities


def k_means(graph: nx.Graph,
            starts: list[int],
            weight: str) -> dict[int, int]:
    adjacency = graph._adj
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    fringe = []
    node2cms = {
        s: i for i, s in enumerate(starts)
    }
    for start in starts:
        push(fringe, (0.0, next(c), 0, start, start))
    while fringe:
        (d, _, n, v, p) = pop(fringe)
        if v in dist:
            continue
        node2cms[v] = node2cms[p]
        dist[v] = (d, n)
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if u not in dist:
                push(fringe, (vu_dist, next(c), n + 1, u, v))
    return node2cms


def validate_cms(
        graph: nx.Graph,
        communities: Community,
        cluster_name: str) -> Community:
    cls = []
    for i, c in enumerate(communities):
        for n in nx.connected_components(graph.subgraph(c)):
            cls.append(n)
    for i, ids in enumerate(cls):
        for j in ids:
            graph.nodes()[j][cluster_name] = i
    return cls
