import networkx as nx


def pfa_heuristic(
        graph: nx.Graph,
        from_node: int,
        to_node: int,
        cluster_adjacency: dict[tuple[int, int], float],
        node_dst: dict[int, float],
        weight: str = "length",
) -> tuple[float, list[int]]:
    nodes = graph.nodes()

    def func(u, v):
        c1 = nodes[u]['cluster']
        c2 = nodes[v]['cluster']
        pi_l = abs(cluster_adjacency[c1, c2] - node_dst[v])
        return abs(pi_l - node_dst[u])

    path = nx.astar_path(graph, from_node, to_node, heuristic=func, weight=weight)
    l = 0
    edges = graph.edges()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        l += edges[u, v][weight]
    return l, path
