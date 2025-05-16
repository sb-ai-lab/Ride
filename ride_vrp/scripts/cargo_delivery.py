import logging
from dataclasses import dataclass
from itertools import groupby
from typing import Optional, List, Dict

import geopy.distance
import networkx as nx

from scripts.data_model import Cargo, Route, Tariff, MergedCargo
from scripts.gis_matrix import get_gis_data
from scripts.multi_knapsack import greedy_merge_cargo
from scripts.routing_model import find_optimal_paths

logger = logging.getLogger('cargo_delivery')

__all__ = [
    'delivery',
    'PathBuildingResult'
]


@dataclass
class PathBuildingResult:
    """
        Класс с информацией о найденном решении.
        :param routes список найденных маршрутов (в нем пути это ноды из графа заказов)
        :param simple_routes упрощенный список найденных маршрутов (в нем пути это просто адреса)
        :param cargo_to_route какие грузы на каких машинах развозятся (вида id груза : id товара)
        :param cargo_graph граф заказов
    """
    routes: List[Route]
    simple_routes: List[Route]
    cargo_to_route: Dict[object, object]
    cargo_graph: nx.Graph


def get_merged_cargos(
        cargos: list[Cargo],
        tariffs: list[Tariff]
) -> list[MergedCargo]:
    """
    Мерж грузов с общими точками A->B в один
    :param cargos: список грузов для мержа
    :param tariffs: список машин
    :return: объединенный грузы
    """
    logger.info('начало мержа грузов')

    def group_key(crg: Cargo):
        return crg.nodes[0], crg.nodes[1]

    merged_cargos: list[MergedCargo] = list(filter(lambda crg: len(crg.nodes) > 2, cargos))
    cargos = list(filter(lambda crg: len(crg.nodes) == 2, cargos))

    for k, v in groupby(sorted(cargos, key=group_key), key=group_key):
        v = list(v)
        res = greedy_merge_cargo(
            cargos=v,
            max_volume=Tariff.DEFAULT_VOLUME_UTILIZATION * max(car.volume for car in tariffs),
            max_mass=Tariff.DEFAULT_MASS_UTILIZATION * max(car.mass for car in tariffs)
        )
        for _, tmp in res.items():
            merged_cargos.append(MergedCargo(
                id=len(merged_cargos),
                mass=[sum(crg.mass[i] for crg in tmp) for i in range(2)],
                volume=[sum(crg.volume[i] for crg in tmp) for i in range(2)],
                nodes=tmp[0].nodes,
                service_time_minutes=[max(crg.service_time_minutes[i] for crg in tmp) for i in range(2)],
                merged_ids=[crg.id for crg in v]
            ))
    logger.info(f'грузы объединены: было {len(cargos)} стало {len(merged_cargos)}')
    return merged_cargos


def generate_graph(
        cargos: list[Cargo],
        points_to_coordinate: dict[object, tuple[float, float]],
        distance_matrix: dict[tuple[object, object], float],
        time_matrix: dict[tuple[object, object], float]
) -> nx.Graph:
    """
    Построение графа заказов.
    Ноды графа это картежи вида (номер заказа, адрес заказа). Каждому заказу соответствует две ноды в графе:
    (номер заказа, начальный адрес) и (номер заказа, конечный адрес) (в будущем начальная и конечная ноды).
    Каждая нода в графе так же хранит информацию:
    - массе и объеме, на которые изменяется вместимость машины, при
    посещении данной надо (либо положительно если груз подбирается, либо отрицательно, если груз скидывается)
    - координаты точки
    - каждая начальная нода хранит флаг, что она начальная и ссылку на конечную.
    - каждая конечная нода хранит ссылку на начальную.

    Между каждой парой нод добавляется ребро, в котором есть следующая информация:
    - длина ребра в метрах

    :param cargos: список грузов
    :param points_to_coordinate: словарь с координатами адресов
    :param distance_matrix: словарь с расстояниями между адресами
    :return:
    """
    logger.info(f'Начало создания графа грузов')
    cargo_graph = nx.DiGraph()
    for d in cargos:
        for i, node in enumerate(d.nodes):
            cargo_graph.add_node(
                (d.id, node, i),
                x=points_to_coordinate[node][0],
                y=points_to_coordinate[node][1],
                start=(i == 0),
                prev_node=None if i == 0 else (d.id, d.nodes[i - 1], i - 1),
                next_node=None if i == len(d.nodes) - 1 else (d.id, d.nodes[i + 1], i + 1),
                mass=d.mass[i],
                volume=d.volume[i],
                service_time=d.service_time_minutes[i]
            )

    for u in cargo_graph.nodes():
        for v in cargo_graph.nodes():
            if u == v:
                continue
            if u[1] == v[1]:
                cargo_graph.add_edge(u, v, length=0, time=0)
            elif (u[1], v[1]) in distance_matrix:
                if distance_matrix[u[1], v[1]] == 0:
                    logger.warning(f'Расстояние между {u[1]} и {v[1]} равно 0.')
                    du = cargo_graph.nodes()[u]
                    dv = cargo_graph.nodes()[v]
                    l = geopy.distance.geodesic((du['x'], du['y']), (dv['x'], dv['y'])).km * 1000
                    cargo_graph.add_edge(u, v, length=l, time=l / 1000)
                else:
                    cargo_graph.add_edge(u, v, length=distance_matrix[u[1], v[1]], time=time_matrix[u[1], v[1]])
            else:
                logger.warning(f'Расстояние между {u[1]} и {v[1]} не задано.')
    logger.info(f'Граф грузов создан: {len(cargo_graph.nodes)} нод и {len(cargo_graph.edges)} ребер')
    return cargo_graph


def get_simple_routes(routes: list[Route]) -> list[Route]:
    """
    Модели удобнее работать с графов заявок и анализ маршрутов удобнее проводить со списком картежей вида
    list[(номер заказа, адрес заказа)]
    Эта функция преобразует маршруты к виду list[адрес]

    :param routes: найденные маршруты
    :return: упрощенные маршруты
    """
    res = []
    for route in routes:
        path = route.path
        r = Route(
            id=route.id,
            # вытаскиваем только адреса
            path=[p[1] for i, p in enumerate(path) if i == len(path) - 1 or path[i + 1][1] != path[i][1]],
            tariff=route.tariff
        )
        res.append(r)

    return res


def get_cargo_to_routes(routes: list[Route], merged_cargos: list[MergedCargo]) -> Dict[object, object]:
    """
    Восстановление какой груз на какой машине будет транспортироваться.
    :param routes: Список построенных маршрутов
    :param merged_cargos: Список грузо
    :return: словарь какой id груза на каком id машины.
    """
    cargo_to_routes = {}
    merged_cargos_ids_2_merged_cargo = {m.id: m for m in merged_cargos}
    for route in routes:
        for p in route.path:
            for crg in merged_cargos_ids_2_merged_cargo[p[0]].merged_ids:
                cargo_to_routes[crg] = route.id
    return cargo_to_routes


def delivery(
        cargos: list[Cargo],
        tariffs: list[Tariff],
        *,
        points_to_coordinate: Optional[dict[object, tuple[float, float]]] = None,
        distance_matrix: Optional[dict[tuple[object, object], float]] = None,
        time_matrix: Optional[dict[tuple[object, object], float]] = None
) -> PathBuildingResult:
    """
    Основная функция модели. Принимает на вход грузы и машины для развоза, мержит грузы и вызывает модель для
    оптимизации маршрутов.

    :param cargos: Список грузов, которые необходимо развести
    :param tariffs: Список машин для развозки
    :param points_to_coordinate: Опциональный словарь для отображения адресов в координаты (если нет, то вызовет 2gis)
    :param distance_matrix: Опциональный словарь для расстояния между адресами (если нет, то вызовет 2gis)
    :param time_matrix: Опциональный словарь для времени между адресами (если нет, то вызовет 2gis)
    :return: PathBuildingResult - результат построения маршрутов
    """

    merged_cargos: list[MergedCargo] = get_merged_cargos(cargos, tariffs)

    points = set()
    for crg in merged_cargos:
        points.update(crg.nodes)

    logger.info(f"количество точек: {len(points)}")

    if not distance_matrix:
        data = get_gis_data(points, coordinates=points_to_coordinate)
        if not points_to_coordinate:
            points_to_coordinate = data.coordinates
        distance_matrix = data.distance_matrix
        time_matrix = data.time_matrix
    cargo_graph = generate_graph(
        merged_cargos,
        points_to_coordinate,
        distance_matrix,
        time_matrix
    )

    routes = find_optimal_paths(cargo_graph, tariffs)
    return PathBuildingResult(
        routes=routes,
        simple_routes=get_simple_routes(routes),
        cargo_to_route=get_cargo_to_routes(routes, merged_cargos),
        cargo_graph=cargo_graph
    )
