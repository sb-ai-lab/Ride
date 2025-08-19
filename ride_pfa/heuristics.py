import logging
from typing import Optional

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

try:
    import kmedoids
except ImportError as e:
    log.error("kmedoids is not installed")
    raise e
try:
    from sklearn.cluster import HDBSCAN
except ImportError as e:
    log.error("sklearn is not installed")
    raise e

__all__ = [
    "angle_clustering",
    "spherical_medoids",
    "radial_clustering"
]


class GraphAttributeError(Exception):
    def __init__(self, attribute, *args):
        super().__init__(args)
        self.attribute = attribute

    def __str__(self):
        return f"Graph node does not contain attribute '{self.attribute}'"


def angle_clustering(dg: nx.Graph,
                     central_node_id: object,
                     end_nodes: Optional[set[object]] = None,
                     min_cluster_size: int = 3) -> None:
    """
    Clusters requests based on their angle w.r.t x axis.\
    Requires the graph node to have coordinate attributes 'x' and 'y'.

    Parameters
    ----------
    dg : nx.Graph
        Graph of deliveries, where each node is eithe a strating point or delivery point.
    central_node_d : int
        Graph id of central node (or hub), which serves as vectors' starting point
    end_nodes : set = None
        Set of ids, to specify end nodes of vectors. By default all nodes in Graph are treated as end nodes
    min_cluster_size: int = 3
        Minimum size of cluster for HDBSCAN

    Returns
    --------
    None
    """
    node = dg.nodes[next(iter(dg.nodes))]
    if "x" not in node:
        raise GraphAttributeError("x")
    if "y" not in node:
        raise GraphAttributeError("y")

    if end_nodes is None:
        end_nodes = dg.nodes
    central_node = dg.nodes[central_node_id]

    xx = [[np.arctan2(d['y'] - central_node['y'], d['x'] - central_node['x'])]
          for u, d in dg.nodes(data=True) if u in end_nodes]
    x = np.array(xx)

    scan = HDBSCAN(min_cluster_size=min_cluster_size)
    membership = scan.fit_predict(x)

    order_2_id = {order: id for order, id in enumerate(end_nodes)}

    for i, label in enumerate(membership):
        dg.nodes[order_2_id[i]]['angle_cluster'] = label

    # Mark nodes, which are located in same area as the central node
    invalid_nodes = [u for u, d in dg.nodes(data=True) if np.allclose(
        [d['x'], d['y']], [central_node['x'], central_node['y']])]
    for id in invalid_nodes:
        dg.nodes[id]['angle_cluster'] = -1


def spherical_medoids(dg: nx.Graph,
                      central_node_id: int,
                      end_nodes: set = None,
                      kmax: int = 10,
                      kmin: int = 2) -> None:
    """
    Clusters requests based on their pairwise angles.\
    Requires the graph node to have coordinate attributes 'x' and 'y'.

    Parameters
    ----------
    dg : nx.Graph
        Graph of deliveries, where each node is eithe a strating point or delivery point.
    central_node_d : int
        Graph id of central node (or hub), which serves as vectors' starting point
    end_nodes : set = None
        Set of ids, to specify end nodes of vectors. By default all nodes in Graph are treated as end nodes
    kmax: int = 10
        Maximum number of medoids to consider
    kmin: int = 2
        Minimum number of medoids to consider

    Returns
    --------
    list[nx.Graph]
        List of clustered subgraphs
    """
    node = dg.nodes[next(iter(dg.nodes))]
    if "x" not in node:
        raise GraphAttributeError("x")
    if "y" not in node:
        raise GraphAttributeError("y")

    if end_nodes is None:
        end_nodes = dg.nodes
    central_node = dg.nodes[central_node_id]

    # Mark nodes, which are located in same area as the central node
    invalid_nodes = [u for u, d in dg.nodes(data=True) if np.allclose(
        [d['x'], d['y']], [central_node['x'], central_node['y']])]
    for id in invalid_nodes:
        dg.nodes[id]['angle_cluster'] = -1

    vectors = np.array([[d['y'] - central_node['y'], d['x'] - central_node['x']]
                        for u, d in dg.nodes(data=True) if u in end_nodes and u not in invalid_nodes])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    d_matrix = np.arccos(vectors @ vectors.T)

    membership = kmedoids.dynmsc(d_matrix, kmax, kmin).labels

    order_2_id = {order: id for order, id in enumerate(end_nodes)}

    for i, label in enumerate(membership):
        dg.nodes[order_2_id[i]]['spherical_cluster'] = label


def radial_clustering(dg: nx.Graph,
                      central_node_id: object,
                      len_attrbute: str = 'length',
                      min_cluster_size: int = 3) -> None:
    """
    Clustering, suitable for deliveries, starting from one place.\
    We assume that all deliveries share starting point, thus we just cluster the\
    length of the deliveries.

    Parameters
    ----------
    dg : nx.Graph
        Graph of deliveries, where each node is eithe a strating point or delivery point.
    central_node_d : int
        Graph id of central node (or hub), which serves as vectors' starting point
    len_attrbute : str = 'length'
        Key word for accessing infromation about the length of the delivery in edge.
    min_cluster_size: int = 3
        Minimum size of cluster for HDBSCAN

    Returns
    --------
    list(int)
        List of clustered deliveries (their ids).
    """
    xx = []
    end_nodes = []

    for node, data in dg._adj[central_node_id].items():
        xx.append(data[len_attrbute])
        end_nodes.append(node)
    x = np.array(xx)

    scan = HDBSCAN(min_cluster_size=min_cluster_size)
    membership = scan.fit_predict(x[:, None])

    order_2_id = {order: id for order, id in enumerate(end_nodes)}

    for i, label in enumerate(membership):
        dg.nodes[order_2_id[i]]['radial_cluster'] = label
