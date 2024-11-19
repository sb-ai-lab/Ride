import networkx as nx
import pandas as pd
import numpy as np
import random
import math
import time
from typing import Dict, Tuple, List
from ride.utils import DataGenerator

class GraphProcessor:
    """
    A class for processing graphs.

    Attributes
    ----------
    None

    Methods
    -------
    create_G_centroid(H)
        Creates a graph consisting of centroids of clusters and returns this graph along with clusters.
    louvain_clusters(H, seed, weight, resolution)
        Applies the Louvain community detection algorithm to the graph.
    create_points_for_test(H, min_distance)
        Generates points for testing based on clusters and distances between them.
    search_resolutions(H, resolution, weight, alpha_max)
        Searches for resolutions for the Louvain algorithm that achieve the desired ratio of the number of clusters to the number of nodes.
    _get_clusters(H)
        Extracts clusters from the graph.
    _get_cluster_transitions(clusters, H)
        Returns a list of neighboring clusters for each cluster.
    _get_centroids(clusters, H)
        Determines the centroids of clusters.
    _create_graph_from_dict(H, cluster_transitions, centroids)
        Creates a new graph based on centroids and transitions between clusters.
    _add_clusters_to_nodes(H, communities)
        Adds information about clusters to nodes in the graph.
    """

    def create_G_centroid(self, H: nx.Graph) -> Tuple[nx.Graph, Dict[int, List[int]]]:
        """
        Creates a graph consisting of centroids of clusters and returns this graph along with clusters.

        Parameters
        ----------
        H : nx.Graph
            The input graph.

        Returns
        -------
        G : nx.Graph
            The graph with centroids.
        clusters : Dict[int, List[int]]
            A dictionary of clusters, where the key is the cluster number and the value is a list of nodes.
        """

        clusters = GraphProcessor._get_clusters(self, H)
        cluster_transitions = GraphProcessor._get_cluster_transitions(self, clusters, H)
        centroids = GraphProcessor._get_centroids(self, clusters, H)
        G = GraphProcessor._create_graph_from_dict(self, H, cluster_transitions, centroids)
        return G, clusters

    def louvain_clusters(self, H: nx.Graph, seed: int = 0, weight: str = 'length', resolution: float = 1) -> Tuple[nx.Graph, Dict[int, List[int]]]:
        """
        Applies the Louvain community detection algorithm to the graph.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        seed : int
            The seed for generating random numbers (default is 0).
        weight : str
            The edge weight (default is 'length').
        resolution : float
            The resolution parameter (default is 1).

        Returns
        -------
        H : nx.Graph
            The graph with added clusters.
        communities : Dict[int, List[int]]
            A dictionary of clusters.
        """

        communities = nx.community.louvain_communities(H, seed=seed, weight=weight, resolution=resolution)
        GraphProcessor._add_clusters_to_nodes(self, H, communities)
        return H, communities

    def create_points_for_test(self, H: nx.Graph, min_distance: int = 10) -> pd.DataFrame:
        """
        Generates points for testing based on clusters and distances between them.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        min_distance : int
            The minimum distance between points (default is 10).

        Returns
        -------
        res : pd.DataFrame
            The points for testing.
        """

        H, _ = GraphProcessor.louvain_clusters(self, H, resolution=1, weight='length')
        G_centroid, clusters = GraphProcessor.create_G_centroid(self, H)
        random.seed(1)
        res = DataGenerator.sample_nodes_for_experiment(self, G_centroid, clusters, min_distance)
        return res

    def search_resolutions(self, H: nx.Graph, resolution: float = 0.001, weight: str = 'length', alpha_max: float = 0.7) -> Tuple[List[float], List[float], List[int], List[int]]:
        """
        Searches for resolutions for the Louvain algorithm, where the desired ratio of the number of clusters to the number of nodes is achieved.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        resolution : float
            The starting resolution for the Louvain algorithm (default is 0.001).
        weight : str
            The edge weight (default is 'length').
        alpha_max : float
            The maximum value of the ratio of the number of clusters to the number of nodes (default is 0.7).

        Returns
        -------
        resolutions : List[float]
            The list of resolutions.
        alphas : List[float]
            The list of ratios.
        nodes_subgraph : List[int]
            The list of the number of nodes in the subgraph.
        edges_subgraph : List[int]
            The list of the number of edges in the subgraph.
        """
        resolutions = []
        alpha = 0
        alphas = []
        nodes_subgraph = []
        edges_subgraph = []

        while alpha < alpha_max:
            H, communities = GraphProcessor.louvain_clusters(self, H, resolution=resolution, weight=weight)
            alpha = len(communities)/len(H.nodes)
            if alpha < 0.008:
                resolution *= 3
                continue
            else:
                G_centroid, _ = GraphProcessor.create_G_centroid(H)
                if len(G_centroid.nodes()) > 0 and len(G_centroid.edges()) > 0:
                    nodes_subgraph.append(len(G_centroid.nodes()))
                    edges_subgraph.append(len(G_centroid.edges()))
                    resolutions.append(resolution)
                    alphas.append(alpha)
                    resolution *= 3
                    resolution = round(resolution,3)

        return resolutions, alphas, nodes_subgraph, edges_subgraph

    def _get_clusters(self, H: nx.Graph) -> Dict[int, List[int]]:
        """
        Extracts clusters from the graph.

        Parameters
        ----------
        H : nx.Graph
            The input graph.

        Returns
        -------
        clusters : Dict[int, List[int]]
            A dictionary of clusters.
        """
        clusters = {}
        for node, data in H.nodes(data=True):
            cluster = data['cluster']
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(node)
        return clusters

    def _get_cluster_transitions(self, clusters: Dict[int, List[int]], H: nx.Graph) -> Dict[int, List[int]]:
        """
        Returns a list of neighboring clusters for each cluster.

        Parameters
        ----------
        clusters : Dict[int, List[int]]
            The clusters of the graph.
        H : nx.Graph
            The input graph.

        Returns
        -------
        cluster_transitions : Dict[int, List[int]]
            A dictionary of cluster transitions.
        """
        cluster_transitions = {}
        for cluster, nodes in clusters.items():
            neighboring_clusters = set()
            for node in nodes:
                for neighbor in H.neighbors(node):
                    neighbor_cluster = H.nodes[neighbor]['cluster']
                    if neighbor_cluster!= cluster:
                        neighboring_clusters.add(neighbor_cluster)
            cluster_transitions[cluster] = list(neighboring_clusters)
        return cluster_transitions

    def _get_centroids(self, clusters: Dict[int, List[int]], H: nx.Graph) -> Dict[int, int]:
        """
        Determines the centroids of clusters.

        Parameters
        ----------
        clusters : Dict[int, List[int]]
            The clusters of the graph.
        H : nx.Graph
            The input graph.

        Returns
        -------
        centroids : Dict[int, int]
            A dictionary of centroids.
        """

        centroids = {}
        for i in range(len(clusters)):
            nodes_ = [node for node, data in H.nodes(data=True) if data.get('cluster') == i]
            s = H.subgraph(nodes_)
            closeness_centrality = nx.closeness_centrality(s)
            centroid_node = max(closeness_centrality, key=closeness_centrality.get)
            centroids[i] = centroid_node
        return centroids
    
    def _create_graph_from_dict(self, H, cluster_transitions: Dict[int, List[int]], centroids: Dict[int, int]) -> nx.Graph:
        """
        Creates a new graph based on centroids and transitions between clusters.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        cluster_transitions : Dict[int, List[int]]
            A dictionary of cluster transitions.
        centroids : Dict[int, int]
            A dictionary of centroids.

        Returns
        -------
        G : nx.Graph
            The new graph.
        """

        G = nx.Graph()
        for node, neighbors in cluster_transitions.items():
            for neighbor in neighbors:
                nodes_ = [n for n, data in H.nodes(data=True) if data.get('cluster') in (node, neighbor)]
                s = H.subgraph(nodes_)
                length, path = nx.single_source_dijkstra(H, centroids[node], centroids[neighbor], weight='length')
                G.add_edge(node, neighbor, weight=length)
        return G

    def _add_clusters_to_nodes(self, H: nx.Graph, communities: Dict[int, List[int]]) -> None:
        """
        Adds information about clusters to nodes in the graph.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        communities : Dict[int, List[int]]
            A dictionary of clusters.
        """
        for i, ids in enumerate(communities):
            for j in ids:
                H.nodes[j]['cluster'] = i

    
class GraphRunner:
    """
    A class for running graph clustering algorithms and evaluating their performance.

    Attributes
    ----------
    None

    Methods
    -------
    test(H, resolutions, k=100, min_distance=10, weight='length')
        Test the performance of the graph clustering algorithm on a given graph.
    compute_shortest_path_length_dijkstra(H, part, weight)
        Compute the shortest path length between the pairs of nodes using Dijkstra's algorithm.
    calculate_error(all_length, all_length_centroids)
        Calculate the error between the shortest path lengths computed using Dijkstra's algorithm and the centroid graph.
    compute_shortest_path_length_centroid(H, resolution, part, weight, output)
        Compute the shortest path length between the pairs of nodes using the centroid graph.
    """

    def test(self, H: nx.Graph, resolutions: list, k: int=100, min_distance: int=10, weight:str='length'):
        """
        Test the performance of the graph clustering algorithm on a given graph.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        resolutions : list
            A list of resolution values to use for the Louvain clustering algorithm.
        k : int, optional
            The number of random pairs of nodes to test (default is 100).
        min_distance : int, optional
            The minimum distance between the nodes in the random pairs (default is 10).
        weight : str, optional
            The name of the edge attribute to use as the weight (default is 'length').

        Returns
        -------
        A pandas DataFrame containing the results of the test.
        """

        # Create a list of random pairs of nodes
        res = GraphProcessor.create_points_for_test(self, H, min_distance=min_distance)

        try:
            part = random.sample(res, k=k)
        except ValueError as ex:
            print(ex)
            part = res

        output = {
            'ks': [],
            'errors_percent': [],
            'time_preprocess_g_centroid_creation': [],
            'time_preprocess_dijkstra_on_centroid_len': [],  
            'time_dijkstra_base': float
        }

        # Compute the shortest path length between the pairs of nodes using Dijkstra's algorithm
        time_base_dijkstra, all_length_dijkstra = GraphRunner.compute_shortest_path_length_dijkstra(self, H, part, weight)
        output['time_dijkstra_base'] = time_base_dijkstra

        # Compute the shortest path length between the pairs of nodes using the centroid graph

        errors_dict = {}
        for c, resolution in enumerate(resolutions):
            output, all_length_centroids = GraphRunner.compute_shortest_path_length_centroid(self, H, resolution, part, weight, output)
            errors = GraphRunner.calculate_error(self, all_length_dijkstra, all_length_centroids)
            output['errors_percent'].append(errors["total_errors"])

            errors_dict[output['ks'][c]] = errors["all_errors"]

        output = pd.DataFrame(output)
        errors_dict = pd.DataFrame(errors_dict)

        output['speedup'] = output['time_dijkstra_base'] / output['time_preprocess_dijkstra_on_centroid_len']

        return output, errors_dict

    def compute_shortest_path_length_dijkstra(self, H: nx.Graph, part: list, weight: str):
        """
        Compute the shortest path length between the pairs of nodes using Dijkstra's algorithm.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        part : list
            A list of random pairs of nodes.
        weight : str
            The name of the edge attribute to use as the weight.

        Returns
        -------
        A list of shortest path lengths between the pairs of nodes.
        """

        start = time.time()
        all_length = []
        for i in part:
            length, path1 = nx.single_source_dijkstra(H, i[0], i[1], weight=weight)
            all_length.append(length)
        end = time.time()
        time_calc = end - start

        return time_calc, all_length
    
    def calculate_error(self, all_length, all_length_centroids):
        """
        Calculate the error between the shortest path lengths computed using Dijkstra's algorithm and the centroid graph.

        Parameters
        ----------
        all_length : list
            A list of shortest path lengths computed using Dijkstra's algorithm.
        all_length_centroids : list
            A list of shortest path lengths computed using the centroid graph.

        Returns
        -------
        A dictionary containing the total error and all errors.
        """

        # errors = (np.array(all_length_centroids).sum() - np.array(all_length).sum()) / np.array(all_length).sum() * 100
        # errors_for_box_plot = (np.array(all_length_centroids) - np.array(all_length)) / np.array(all_length) * 100
        errors = (np.array(all_length_centroids).sum() * 100 / np.array(all_length).sum()) - 100
        errors_for_box_plot = (np.array(all_length_centroids) * 100 / np.array(all_length)) - 100
        return {"total_errors": errors, "all_errors": errors_for_box_plot}
        

    def compute_shortest_path_length_centroid(self, H: nx.Graph, resolution: int, part: list, weight: str, output):
        """
        Compute the shortest path length between the pairs of nodes using the centroid graph.

        Parameters
        ----------
        H : nx.Graph
            The input graph.
        resolution : int
            Resolution value to use for the Louvain clustering algorithm.
        part : list
            A list of random pairs of nodes.
        weight : str
            The name of the edge attribute to use as the weight.
        output : dict
            A dictionary to store the results.

        Returns
        -------
        A pandas DataFrame containing the results of the test.
        """

        v1 = []
        e1  =[]
        H, communities = GraphProcessor.louvain_clusters(self, H, resolution=resolution, weight=weight)
        k = len(communities)/len(H.nodes)
        output['ks'].append(round(k, 3))

        centroid_g_time_creation_start = time.time()
        G_centroid, clusters = GraphProcessor.create_G_centroid(self, H)
        centroid_g_time_creation_end = time.time() - centroid_g_time_creation_start

        output['time_preprocess_g_centroid_creation'].append(centroid_g_time_creation_end)
        v1.append(len(G_centroid.nodes()))
        e1.append(len(G_centroid.edges()))

        all_l_c = []
        start = time.time()
        for i in part:
            cluster_1 = H.nodes(data=True)[i[0]]['cluster']
            cluster_2 = H.nodes(data=True)[i[1]]['cluster']
            leng_G = nx.dijkstra_path(G_centroid, cluster_1, cluster_2)
            nodes = []
            for g_name in leng_G:
                nodes.extend(clusters[g_name])
            Hs = H.subgraph(nodes)
            length, _ = nx.single_source_dijkstra(Hs, i[0], i[1], weight='length')

            all_l_c.append(length)

        result = time.time() - start
        output['time_preprocess_dijkstra_on_centroid_len'].append(result)

        return output, all_l_c
    
if __name__ == "__main__":
    # Example usage of the GraphProcessor class
    graph = nx.erdos_renyi_graph(100, 0.05)
    gp = GraphProcessor()
    G, clusters = gp.louvain_clusters(graph)
    print("Number of clusters:", len(clusters))
    print("Number of nodes in G:", G.number_of_nodes())
    print("Number of edges in G:", G.number_of_edges())