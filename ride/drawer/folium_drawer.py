import random
from typing import Optional

import folium
import networkx as nx

from ride.clustering import Community

__all__ = [
    'draw_graph',
    'draw_path'
]


def draw_graph(
        graph: nx.Graph,
        cms: Optional[Community] = None,
        m: Optional[folium.Map] = None) -> folium.Map:
    if not m:
        x, y = 0, 0
        for _, d in graph.nodes(data=True):
            x, y = d['x'], d['y']
            break
        m = folium.Map(location=[y, x], zoom_start=12)
    if cms is not None:
        colors = generate_colors(len(cms))
        for i, c in cms:
            for u in c:
                d = graph.nodes()[u]
                folium.Circle(
                    [d['y'], d['x']],
                    popup=str(u),
                    fill=True,
                    color=colors[i]
                ).add_to(m)
    else:
        for node in graph.nodes():
            folium.Circle(
                [graph.nodes[node]['y'], graph.nodes[node]['x']],
                popup=str(node),
                fill=True
            ).add_to(m)

    for edge in graph.edges():
        folium.PolyLine(
            [[graph.nodes[edge[0]]['y'], graph.nodes[edge[0]]['x']],
             [graph.nodes[edge[1]]['y'], graph.nodes[edge[1]]['x']]],
            color='blue',
            weight=2
        ).add_to(m)

    return m


def draw_path(graph: nx.Graph, path: list[int],
              m: Optional[folium.Map],
              color: str = 'red',
              weight: float = 5) -> folium.Map:
    x = [graph.nodes[node]['x'] for node in path]
    y = [graph.nodes[node]['y'] for node in path]
    if not m:
        m = folium.Map(location=[y[0], x[0]], zoom_start=12)

    for i in range(len(path) - 1):
        folium.PolyLine([[y[i], x[i]], [y[i + 1], x[i + 1]]], color=color, weight=weight).add_to(m)

    return m


def random_color_hex():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def generate_colors(num_colors):
    return [random_color_hex() for _ in range(num_colors)]
