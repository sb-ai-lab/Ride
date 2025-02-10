import random
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ride_pfa.clustering import Community

__all__ = [
    'draw_graph_matplotlib',
    'draw_paths_matplotlib'
]

def draw_graph_matplotlib(graph: nx.Graph, cms: Optional[Community] = None, ax: Optional[plt.Axes] = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 20))

    pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in graph.nodes()}
    
    if cms is not None:
        colors = generate_colors(len(cms))
        for i, c in cms:
            nx.draw_networkx_nodes(graph, pos, nodelist=c, node_color=colors[i], ax=ax, node_size=10)
    else:
        nx.draw_networkx_nodes(graph, pos, node_color='blue', ax=ax, node_size=10)

    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=1, ax=ax)
    
    ax.set_title("Graph Visualization")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return ax


def draw_paths_matplotlib(graph: nx.Graph, paths: list[tuple[list[int], str]], ax: Optional[plt.Axes] = None):
    """
    Рисует граф и список путей на одном изображении.

    :param graph: Граф (networkx)
    :param paths: Список кортежей (путь, цвет), где путь — это список узлов, цвет — цвет линии пути.
                  Если цвет не задан, он будет выбран автоматически.
    :param ax: График matplotlib (если None, создается новый)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 20))

    pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in graph.nodes()}

    # Рисуем граф один раз
    draw_graph_matplotlib(graph, ax=ax)

    # Генерируем цвета, если они не заданы
    colors = generate_colors(len(paths))

    legend_patches = []
    for i, (path, color) in enumerate(paths):
        color = color if color else colors[i]

        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color=color, width=3, ax=ax)

        # Вычисляем длину пути
        total_length = sum(graph.edges[u, v].get('length', 0) for u, v in path_edges)

        # Добавляем элемент в легенду
        legend_patches.append(mpatches.Patch(color=color, label=f"Path {i+1}: {total_length:.2f}"))

    # Добавляем легенду
    ax.legend(handles=legend_patches, loc="upper right")

    ax.set_title("Graph with Paths")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return ax


def random_color_hex():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def generate_colors(num_colors):
    return [random_color_hex() for _ in range(num_colors)]