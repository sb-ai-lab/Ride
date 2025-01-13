from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

import networkx as nx


@dataclass
class PathFindingAdvanced:
    g: nx.Graph
    weight: str = 'length'

    def find_distance_to_all(self,
                             start: int,
                             ends: set[int]) -> dict[int, float]:

        weight: str = self.weight
        graph = self.g
        adjacency = graph._adj
        c = count()
        push = heappush
        pop = heappop
        dist = {}
        visited = set()
        fringe = []
        push(fringe, (0.0, next(c), start))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue
            dist[v] = d
            if v in ends:
                visited.add(v)
            if len(visited) == len(ends):
                break
            for u, e in adjacency[v].items():
                vu_dist = d + e[weight]
                if u not in dist:
                    push(fringe, (vu_dist, next(c), u))
        return {k: dist[k] for k in ends}
