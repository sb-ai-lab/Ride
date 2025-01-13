import os
import pickle
from typing import Optional

import networkx as nx
import osmnx as ox

__all__ = [
    "get_graph",
    "osm_cities_example"
]

osm_cities_example = {
    'Barcelona': 'R347950',
    'Paris': 'R71525',
    'Prague': 'R435514',
    'Moscow': 'R2555133',
    'Singapore': 'R17140517',
    'Berlin': 'R62422',
    'Rome': 'R41485',
    'Rio': 'R2697338',
    'Delhi': 'R1942586'
}


# load graph
def get_graph(city_id: str = 'R2555133', cache_path: Optional[str] = None) -> nx.Graph:
    id_graph = city_id
    name = f'{id_graph}.pickle'
    if cache_path is not None:
        path = os.path.join(cache_path, name)
        if os.path.exists(path):
            with open(path, 'rb') as fp:
                g: nx.Graph = pickle.load(fp)
                fp.close()
        else:
            g = _get_gr(id_graph)
            with open(path, 'wb') as fp:
                pickle.dump(g, fp)
                fp.close()
    else:
        g = _get_gr(id_graph)
    assert g is not None
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def _get_gr(city_id):
    gdf = ox.geocode_to_gdf(city_id, by_osmid=True)
    polygon_boundary = gdf.unary_union
    graph = ox.graph_from_polygon(polygon_boundary,
                                  network_type='drive',
                                  simplify=True)
    H = nx.Graph()
    # add edges in new graph. Copy only weights
    for u, d in graph.nodes(data=True):
        H.add_node(u, x=d['x'], y=d['y'])
    for u, v, d in graph.edges(data=True):
        if u == v:
            continue
        H.add_edge(u, v, length=d['length'])
    del city_id, gdf, polygon_boundary, graph
    return H
