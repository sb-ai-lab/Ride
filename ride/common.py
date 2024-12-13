# import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import random
from pyproj import Transformer
from shapely.geometry import Point, LineString
from tqdm import tqdm

from ride import pfa
from ride.file_name_generator import generate_new_name
from ride.utils import extract_cluster_list_subgraph


def find_path(
        layer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> tuple[float, list[int]]:
    """
    Find the shortest path between two nodes in a graph using a specified algorithm.

    Parameters
    ----------
    layer : GraphLayer
        The graph layer to search in.
    from_node : int
        The node to start the search from.
    to_node : int
        The node to search for.
    alg : str, optional
        The algorithm to use for the search (default is 'dijkstra'). Can be 'dijkstra', 'bidirectional', or 'astar'.

    Returns
    -------
    tuple[float, list[int]]
        A tuple containing the length of the shortest path and the path itself.

    Raises
    ------
    nx.NetworkXNoPath
        If no path is found between the nodes.
    """
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    if from_cluster == to_cluster:
        g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
        if alg == 'dijkstra':
            return nx.single_source_dijkstra(g, from_node, to_node, weight='length')
        if alg == 'bidirectional':
            return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    g = layer.centroids_graph
    path = []
    if alg == 'dijkstra':
        path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')
    if alg == 'bidirectional':
        path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')

    cls = set()
    cls.add(to_cluster)
    for u in path[1]:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)

    if alg == 'dijkstra':
        path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')
    if alg == 'bidirectional':
        path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')
    return path


def find_path_length(
        layer,
        from_node: int,
        to_node: int,
        alg='dijkstra') -> float:
    """
    Find the length of the shortest path between two nodes in a graph using a specified algorithm.

    Parameters
    ----------
    layer : GraphLayer
        The graph layer to search in.
    from_node : int
        The node to start the search from.
    to_node : int
        The node to search for.
    alg : str, optional
        The algorithm to use for the search (default is 'dijkstra'). Can be 'dijkstra', 'bidirectional', or 'astar'.

    Returns
    -------
    tuple[float, float, float, float]
        A tuple containing the length of the shortest path, and the times taken for each step of the algorithm.

    Raises
    ------
    nx.NetworkXNoPath
        If no path is found between the nodes.
    """
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    if from_cluster == to_cluster:
        g = extract_cluster_list_subgraph(layer.graph, [to_cluster], layer.communities)
        if alg == 'dijkstra':
            return nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
        if alg == 'bidirectional':
            return nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    g = layer.centroids_graph
    path = []
    if alg == 'dijkstra':
        path = nx.single_source_dijkstra(g, from_center, to_center, weight='length')[1]
    if alg == 'bidirectional':
        path = nx.bidirectional_dijkstra(g, from_center, to_center, weight='length')[1]

    cls2next = {}
    cls = set()
    cls.add(to_cluster)
    prev = from_cluster
    for u in path:
        du = layer.graph.nodes[u]
        c = du['cluster']
        if c != prev:
            cls2next[prev] = u
        prev = c
        cls.add(c)
    cls2next[prev] = to_node

    g = extract_cluster_list_subgraph(layer.graph, cls, layer.communities)

    if alg == 'dijkstra':
        path = nx.single_source_dijkstra(g, from_node, to_node, weight='length')[0]
    if alg == 'bidirectional':
        path = nx.bidirectional_dijkstra(g, from_node, to_node, weight='length')[0]
    return path


class CentroidResult:
    """
        Represents the result of a centroid calculation.

        Attributes
        ----------
        resolution : float
            The resolution of the centroid calculation
        centroid_nodes : int
            The number of nodes in the centroid
        centroid_edges : int
            The number of edges in the centroid
        centroid_density : float
            The density of the centroid (calculated as 2 * centroid_edges / (centroid_nodes * (centroid_nodes - 1)))
        alpha : float
            Еhe ratio of the number of clusters to the number of nodes
        speed_up : list[float]
            The acceleration relative to the basic solution
        errors : list[float]
            The error values relative to the basic solution
        absolute_time : list[float]
            The absolute time values for the centroid calculation
        absolute_err : list[float]
            The absolute error values for the centroid calculation

        Parameters
        ----------
        resolution : float
            The resolution of the centroid calculation
        centroid_nodes : int
            The number of nodes in the centroid
        centroid_edges : int
            The number of edges in the centroid
        alpha : float
            Еhe ratio of the number of clusters to the number of nodes
    """

    def __init__(self, resolution: float,
                 centroid_nodes: int,
                 centroid_edges: int,
                 alpha: float
                 ):
        self.resolution: float = resolution
        self.centroid_nodes: int = centroid_nodes
        self.centroid_edges: int = centroid_edges
        if centroid_nodes == 1:
            self.centroid_density = 0
        else:
            self.centroid_density = 2 * centroid_edges / (centroid_nodes * (centroid_nodes - 1))
        self.alpha: float = alpha
        self.speed_up: list[float] = []
        self.errors: list[float] = []
        self.absolute_time: list[float] = []
        self.absolute_err: list[float] = []


class CityResult:
    """
        Represents the result of a city calculation.

        Attributes
        ----------
        name : str
            The name of the city
        name_suffix : str
            The suffix of the city name
        id : str
            The ID of the city
        nodes : int
            The number of nodes in the city graph
        edges : int
            The number of edges in the city graph
        density : float
            The density of the city graph (calculated as 2 * edges / (nodes * (nodes - 1)))
        points_results : list[CentroidResult]
            The list of centroid results for the city

        Parameters
        ----------
        name : str
            The name of the city
        name_suffix : str
            The suffix of the city name
        city_id : str
            The ID of the city
        nodes : int
            The number of nodes in the city graph
        edges : int
            The number of edges in the city graph

        Methods
        -------
        save()
            Saves the city result to a file
    """

    def __init__(self,
                 name: str,
                 name_suffix: str,
                 city_id: str,
                 nodes: int,
                 edges: int
                 ):
        self.name = name
        self.name_suffix: str = name_suffix
        self.id: str = city_id
        self.nodes: int = nodes
        self.edges: int = edges
        self.density: float = 2 * edges / (nodes * (nodes - 1))
        self.points_results: list[CentroidResult] = []

    def save(self):
        file_name = generate_new_name(self.name + self.name_suffix + '.pkl')
        with open(file_name, 'wb') as fp:
            pickle.dump(self, fp)
            fp.close()


class GraphLayer:
    """
        Represents a graph layer with communities, cluster information, and centroids.

        Attributes
        ----------
        cluster_to_center : dict[int, int]
            Mapping of cluster IDs to their corresponding center nodes
        cluster_to_bridge_points : dict[int, set[int]]
            Mapping of cluster IDs to their corresponding bridge points
        cluster_to_neighboring_cluster : dict[int, set[int]]
            Mapping of cluster IDs to their neighboring clusters
        communities : list[set[int]]
            List of communities in the graph
        centroids_graph : nx.Graph
            Graph of centroids
        graph : nx.Graph
            Original graph

        Parameters
        ----------
        graph : nx.Graph
            Original graph
        communities : list[set[int]]
            List of communities in the graph
        cluster_to_neighboring_cluster : dict[int, set[int]]
            Mapping of cluster IDs to their neighboring clusters
        cluster_to_bridge_points : dict[int, set[int]]
            Mapping of cluster IDs to their bridge points
        cluster_to_center : dict[int, int]
            Mapping of cluster IDs to their corresponding center nodes
        centroids_graph : nx.Graph
            Graph of centroids
    """

    def __init__(self,
                 graph: nx.Graph,
                 communities: list[set[int]],
                 cluster_to_neighboring_cluster: dict[int, set[int]],
                 cluster_to_bridge_points: dict[int, set[int]],
                 cluster_to_center: dict[int, int],
                 centroids_graph: nx.Graph
                 ):
        self.cluster_to_center = cluster_to_center
        self.cluster_to_bridge_points = cluster_to_bridge_points
        self.cluster_to_neighboring_cluster = cluster_to_neighboring_cluster
        self.communities = communities
        self.centroids_graph = centroids_graph
        self.graph = graph
        self.cluster_adjacency, self.node_dst = self._compute()

    def draw_graph(self, visible=False):
        m = folium.Map(zoom_start=12)

        for node in self.graph.nodes():
            folium.Circle([self.graph.nodes[node]['y'], self.graph.nodes[node]['x']], popup=str(node),
                          fill=True).add_to(m)
        for edge in self.graph.edges():
            folium.PolyLine([[self.graph.nodes[edge[0]]['y'], self.graph.nodes[edge[0]]['x']],
                             [self.graph.nodes[edge[1]]['y'], self.graph.nodes[edge[1]]['x']]], color='blue',
                            weight=2).add_to(m)

        m.save('graph.html')

        if visible:
            return m

    def draw_path(self, nodes: list[int], x: list[float] = None, y: list[float] = None, visible=False, save=False):
        if not x:
            x = [self.graph.nodes[node]['x'] for node in nodes]
        if not y:
            y = [self.graph.nodes[node]['y'] for node in nodes]

        m = folium.Map(location=[y[0], x[0]], zoom_start=12)

        for i in range(len(nodes) - 1):
            folium.PolyLine([[y[i], x[i]], [y[i + 1], x[i + 1]]], color='red', weight=5).add_to(m)

        if save:
            m.save('path.html')

        if visible:
            return m

    def draw_paths_ctx(self, from_node: int, to_node: int, alg: str = 'dijkstra', weight: str = 'length', visible=False,
                       save=False):
        distance, nodes = self.find_path(from_node, to_node, alg)
        distance_original, path_original = self.find_path_original_graph(from_node, to_node, weight)

        x = [self.graph.nodes[node]['x'] for node in nodes]
        y = [self.graph.nodes[node]['y'] for node in nodes]
        x_original = [self.graph.nodes[node]['x'] for node in path_original]
        y_original = [self.graph.nodes[node]['y'] for node in path_original]

        # Создаем GeoDataFrame для путей
        gdf_custom = gpd.GeoDataFrame(geometry=[LineString(zip(x, y))])
        gdf_original = gpd.GeoDataFrame(geometry=[LineString(zip(x_original, y_original))])

        # Преобразуем координаты из EPSG:4326 в EPSG:3857
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Применяем трансформацию к каждой координате в LineString
        gdf_custom['geometry'] = gdf_custom['geometry'].apply(
            lambda geom: LineString([transformer.transform(xy[0], xy[1]) for xy in geom.coords])
        )

        gdf_original['geometry'] = gdf_original['geometry'].apply(
            lambda geom: LineString([transformer.transform(xy[0], xy[1]) for xy in geom.coords])
        )

        # Отображаем данные на карте
        ax = gdf_custom.plot(color='red', figsize=(10, 10), label=f'Custom Path (Distance: {distance / 1000:.2f} km)')
        gdf_original.plot(ax=ax, color='blue', label=f'Original Path (Distance: {distance_original / 1000:.2f} km)')

        # Добавляем подложку с картой
        # ctx.add_basemap(ax, zoom=12, source=ctx.providers.OpenStreetMap.Mapnik)

        # Убираем оси
        ax.set_axis_off()

        # Добавляем легенду
        ax.legend(loc='upper right')

        # Сохраняем или показываем результат
        if save:
            ax.figure.savefig('paths_ctx.png')

        if visible:
            plt.show()

    def draw_paths_folium(self, from_node: int, to_node: int, alg: str = 'dijkstra', weight: str = 'length',
                          visible=False, save=False):
        distance, nodes = self.find_path(from_node, to_node, alg)
        distance_original, path_original = self.find_path_original_graph(from_node, to_node, weight)

        x = [self.graph.nodes[node]['x'] for node in nodes]
        y = [self.graph.nodes[node]['y'] for node in nodes]
        x_original = [self.graph.nodes[node]['x'] for node in path_original]
        y_original = [self.graph.nodes[node]['y'] for node in path_original]

        m = folium.Map(location=[y[0], x[0]], zoom_start=12)

        # Создаем группы слоев
        layer_original = folium.FeatureGroup(name='Оригинальный путь', color='blue')
        layer_custom = folium.FeatureGroup(name='Путь по алгоритму', color='red')

        # Добавляем линии в соответствующие группы слоев
        for i in range(len(nodes) - 1):
            folium.PolyLine([[y[i], x[i]], [y[i + 1], x[i + 1]]], color='red', weight=5).add_to(layer_custom)
        for i in range(len(path_original) - 1):
            folium.PolyLine([[y_original[i], x_original[i]], [y_original[i + 1], x_original[i + 1]]], color='blue',
                            weight=5).add_to(layer_original)

        # Добавляем группы слоев на карту
        layer_custom.add_to(m)
        layer_original.add_to(m)

        # Добавляем легенду
        folium.LayerControl().add_to(m)

        if save:
            m.save('paths_folium.html')

        if visible:
            return m

    def find_path(self, from_node: int, to_node: int, alg: str = 'heuristic', draw_path=False, visible=False,
                  save=False) -> tuple[float, list[int]]:
        if alg == 'heuristic':
            distance, nodes = pfa.pfa_heuristic(self.graph, from_node, to_node, self.cluster_adjacency, self.node_dst)
        else:
            distance, nodes = find_path(self, from_node, to_node, alg)
        if draw_path:
            x = [self.graph.nodes[node]['x'] for node in nodes]
            y = [self.graph.nodes[node]['y'] for node in nodes]
            maps = self.draw_path(nodes, x, y, visible=visible, save=save)
            return distance, nodes, maps
        return distance, nodes

    def find_path_original_graph(self, from_node: int, to_node: int, weight: str = 'length') -> tuple[float, list[int]]:
        distance, path = nx.single_source_dijkstra(self.graph, from_node, to_node, weight=weight)
        return distance, path

    def draw_clusters(self, markersize=5, alpha=0.7, visible=False, save=False):
        """
        Отображает точки каждого кластера на карте с уникальными цветами для каждого кластера.

        Parameters:
        -----------
        visible : bool, optional
            Если True, отображает карту.
        save : bool, optional
            Если True, сохраняет изображение карты.
        """

        points = []
        colors = []

        # Преобразование координат EPSG:4326 в EPSG:3857
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Генерация цветов для каждого кластера
        cluster_colors = self.generate_colors(len(self.communities))

        for idx, community in enumerate(self.communities):
            # Получаем координаты x и y для всех узлов в сообществе
            for node in community:
                x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
                x_3857, y_3857 = transformer.transform(x, y)  # Преобразуем координаты в EPSG:3857
                points.append(Point(x_3857, y_3857))
                colors.append(cluster_colors[idx])  # Добавляем цвет для каждой точки

        # Создаем GeoDataFrame для точек
        gdf_points = gpd.GeoDataFrame(geometry=points)

        # Отображаем точки
        ax = gdf_points.plot(marker='o', color=colors, markersize=markersize, figsize=(10, 10), alpha=alpha)

        # Добавляем подложку карты
        # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Убираем оси
        ax.set_axis_off()

        # Сохраняем или показываем карту
        if save:
            plt.savefig("cluster_points.png")

        if visible:
            plt.show()

    def random_color_hex(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def generate_colors(self, num_colors):
        return [self.random_color_hex() for _ in range(num_colors)]

    def draw_path_through_clusters(self, from_node: int, to_node: int, alg: str = 'dijkstra', markersize=5, alpha=0.7,
                                   visible=False, save=False):
        """
        Находит путь между двумя точками и отображает только те точки и кластеры, через которые проходит маршрут.

        Parameters:
        -----------
        from_node : int
            Стартовый узел.
        to_node : int
            Конечный узел.
        alg : str, optional
            Алгоритм для поиска маршрута (по умолчанию 'dijkstra').
        markersize : int, optional
            Размер маркеров для точек.
        visible : bool, optional
            Если True, отображает карту.
        save : bool, optional
            Если True, сохраняет изображение карты.
        """

        # Шаг 1: Находим путь между двумя точками
        distance, nodes = self.find_path(from_node, to_node, alg)

        # Шаг 2: Определяем кластеры, через которые проходит маршрут
        clusters_passed = set(self.graph.nodes[node]['cluster'] for node in nodes)

        points = []
        colors = []

        # Преобразование координат EPSG:4326 в EPSG:3857
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Генерация цветов для кластеров, через которые прошел маршрут
        cluster_colors = self.generate_colors(len(clusters_passed))
        cluster_color_map = {cluster: color for cluster, color in zip(clusters_passed, cluster_colors)}

        # Шаг 3: Отображаем только точки тех кластеров, через которые прошел маршрут
        for cluster in clusters_passed:
            for node in self.communities[cluster]:
                x, y = self.graph.nodes[node]['x'], self.graph.nodes[node]['y']
                x_3857, y_3857 = transformer.transform(x, y)  # Преобразуем координаты в EPSG:3857
                points.append(Point(x_3857, y_3857))
                colors.append(cluster_color_map[cluster])  # Добавляем цвет для каждой точки

        # Создаем GeoDataFrame для точек
        gdf_points = gpd.GeoDataFrame(geometry=points)

        # Отображаем точки
        ax = gdf_points.plot(marker='o', color=colors, markersize=markersize, figsize=(10, 10), alpha=alpha)

        # Шаг 4: Отображаем сам маршрут
        path_x = [self.graph.nodes[node]['x'] for node in nodes]
        path_y = [self.graph.nodes[node]['y'] for node in nodes]

        # Преобразуем координаты маршрута
        path_coords = [transformer.transform(x, y) for x, y in zip(path_x, path_y)]

        # Создаем GeoDataFrame для маршрута
        gdf_path = gpd.GeoDataFrame(geometry=[LineString(path_coords)])

        # Отображаем маршрут поверх точек
        gdf_path.plot(ax=ax, color='red', linewidth=2, label=f'Маршрут {alg} (длина: {distance:.2f})')

        # Добавляем подложку карты
        # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Убираем оси
        ax.set_axis_off()

        # Добавляем легенду
        plt.legend()

        # Сохраняем или показываем карту
        if save:
            plt.savefig("path_through_clusters.png")

        if visible:
            plt.show()

    def _compute(self):
        dst = dict(nx.all_pairs_dijkstra_path_length(self.centroids_graph, weight='length'))
        d_clusters = {}
        d_nodes = {}
        for u in tqdm(dst):
            for v in dst[u]:
                c1, c2 = self.graph.nodes[u]['cluster'], self.graph.nodes[v]['cluster']
                d_clusters[c1, c2] = dst[u][v]
                d_clusters[c2, c1] = dst[u][v]

        for u, d in tqdm(self.graph.nodes(data=True)):
            c = self.cluster_to_center[d['cluster']]
            d_nodes[u] = nx.single_source_dijkstra(self.graph, u, c, weight='length')[0]
        return d_clusters, d_nodes
