from dataclasses import dataclass

import networkx as nx
from tqdm.auto import tqdm

from ride import utils
from ride.clustering import AbstractCommunityResolver, Community

__all__ = [
    "CentroidGraph",
    "CentroidGraphBuilder"
]


@dataclass
class CentroidGraph:
    g: nx.Graph
    cls2n: dict[int, set[int]]
    cls2c: dict[int, int]
    cls2hubs: dict[int, set[int]]
    cms: Community


@dataclass
class CentroidGraphBuilder:
    log: bool = False,
    name: str = 'cluster'
    weight: str = 'length'

    def build(self, g: nx.Graph, cms: AbstractCommunityResolver | Community) -> CentroidGraph:
        if isinstance(cms, AbstractCommunityResolver):
            cms = cms.resolve(g)
        cls2n = get_cls2n(g, name=self.name)
        g1, cls2c = build_center_graph(
            graph=g,
            communities=cms,
            cls2n=cls2n,

            log=self.log,
            name=self.name,
            weight=self.weight
        )
        cg = CentroidGraph(
            g=g1,
            cls2n=cls2n,
            cls2c=cls2c,
            cls2hubs=get_cls2hubs(g, name=self.name),
            cms=cms
        )
        return cg

    def build_with_time(self, g: nx.Graph, cms: AbstractCommunityResolver | Community, iterations=1) -> (
            tuple)[float, CentroidGraph]:
        return utils.compute(lambda: self.build(g, cms), iterations=iterations)


# cluster to neighboring clusters
def get_cls2n(graph: nx.Graph, name='cluster') -> dict[int: set[int]]:
    _cls2n = {}
    for u, du in graph.nodes(data=True):
        for v in graph[u]:
            dv = graph.nodes()[v]
            if dv[name] == du[name]:
                continue
            c1 = dv[name]
            c2 = du[name]
            if not (c1 in _cls2n):
                _cls2n[c1] = set()
            if not (c2 in _cls2n):
                _cls2n[c2] = set()
            _cls2n[c1].add(c2)
            _cls2n[c2].add(c1)
    return _cls2n


# cluster then yts point that are connected with neighboring clusters
def get_cls2hubs(graph: nx.Graph, name='cluster') -> dict[int: set[int]]:
    _cls2hubs = {}
    for u, du in graph.nodes(data=True):
        for v in graph[u]:
            dv = graph.nodes()[v]
            c1 = du[name]
            c2 = dv[name]
            if c1 == c2:
                continue
            if not (c1 in _cls2hubs):
                _cls2hubs[c1] = set()
            if not (c2 in _cls2hubs):
                _cls2hubs[c2] = set()
            _cls2hubs[c1].add(u)
            _cls2hubs[c2].add(v)
    return _cls2hubs


# build_center_graph
def build_center_graph(
        graph: nx.Graph,
        communities: Community,
        cls2n: dict[int: set[int]],
        log: bool = False,
        name: str = 'cluster',
        weight: str = 'length'
) -> tuple[nx.Graph, dict[int, int]]:
    x_graph = nx.Graph()
    cls2c = {}
    _iter = tqdm(enumerate(communities), total=len(communities), desc='find centroids') if log else enumerate(
        communities)
    for cls, _ in _iter:
        gc = extract_cluster_list_subgraph(graph, [cls], communities)
        min_node = nx.barycenter(gc, weight=weight)[0]
        du = graph.nodes()[min_node]
        x_graph.add_node(graph.nodes()[min_node][name], **du)
        cls2c[graph.nodes()[min_node][name]] = min_node

    if len(x_graph.nodes) == 1:
        return x_graph, cls2c
    _iter = tqdm(x_graph.nodes(), desc='find edges') if log else x_graph.nodes()
    for u in _iter:
        # todo find path from point to set points
        for v in cls2n[u]:
            length = nx.single_source_dijkstra(graph, source=cls2c[u], target=cls2c[v], weight=weight)[0]
            x_graph.add_edge(u, v, length=length)
    return x_graph, cls2c


# extract subgraph by clusters
def extract_cluster_list_subgraph(graph: nx.Graph, cluster_number: list[int] | set[int], communities=None,
                                  cluster: str = 'cluster') -> nx.Graph:
    if communities:
        return graph.subgraph(_iter_cms(cluster_number, communities))
    else:
        nodes_to_keep = [node for node, data in graph.nodes(data=True) if data[cluster] in cluster_number]
    return graph.subgraph(nodes_to_keep)


def _iter_cms(cluster_number: list[int] | set[int], communities: list[set[int]] | tuple[set[int]]):
    for cls in cluster_number:
        for u in communities[cls]:
            yield u
