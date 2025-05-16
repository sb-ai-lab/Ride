import logging
import random
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from threading import Lock
from typing import Optional

import networkx as nx
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.util.optional_boolean_pb2 import BOOL_FALSE, BOOL_TRUE

from scripts.data_model import Route, TariffCost, Tariff

__all__ = [
    'find_optimal_paths'
]

from scripts.exceptions import CargoLoopException

logger = logging.getLogger(name='routing_model')
"""
Максимальное время работы модели. В худшем случае может не найти решение за это время.
"""
DEFAULT_MINUTES_FOR_MODEL = 5
"""
максимальное кол-во точек на маршруте.
"""
MAX_POINT_NUMBER = 20
"""
Верхняя граница расстояния для машины в метрах
"""
CAR_DISTANCE_UPPER_BOUND = int(1e6)
"""
Верхняя граница времени для машины в минутах
"""
CAR_TIME_UPPER_BOUND = 8 * 60


@dataclass(
    slots=True
)
class Car:
    id: object = field(hash=True)  # идентификатор машины
    tariff_id: object = field(hash=True)  # тариф машины
    max_mass: int = field(hash=False)  # вместимость по массе
    max_volume: int = field(hash=False)  # вместимость по массе
    tariff_cost: TariffCost = field()  # цена и километраж тарифа


@dataclass(
    slots=True
)
class ProblemData:
    """
        Data class для описания проблемы,
    """
    """
        Граф заказов
    """
    graph: nx.Graph
    """
        Отображение вершины графа в номер
    """
    node2index: dict[object, int]
    """
        Отображение  номера вершины в вершину
    """
    index2node: dict[int, object]
    """
        Число вершин в графе
    """
    N: int
    """
        Список доступных машин
    """
    cars: list[Car]


@dataclass(
    init=False,
    slots=True
)
class ModelData:
    # коэфф для масштабирования объема грузов
    VOLUME_SCALE = 100
    """
    Data class с данными для работы модели,
    """
    """
        Описание проблемы.
    """
    problem_data: ProblemData
    """
        Матрица расстояний.
    """
    dsts: np.array
    """
        Количество машин
    """
    num_vehicles: int
    """
        Список списков с порядком посещения, например,
        [[1,2,3],[4,5]] означает, что узлы 1,2,3 должны посещаться одной машиной в указанном порядке.
        4,5 так же должны посещаться одной машиной (либо той же, либо другой) в указанном порядке.    
    """
    pickups_deliveries: list[list]
    """
        Список спроса по массе (либо +, либо - масса, смотря погрузка или разгрузка). 
        Для 0 точки (псевдо депо) спрос всегда 0
        Масса всегда целое число
    """
    demands_mass: list[int]
    """
        Список спроса по объему (либо +, либо - объем, смотря погрузка или разгрузка). 
        Для 0 точки (псевдо депо) спрос всегда 0
        Объем всегда целое число, вычисленное по формуле V_int = int(VOLUME_SCALE * V)  
    """
    demands_volume: list[int]
    """
        Вместимость машины по массе. (масса должна быть целым числом)
    """
    vehicle_capacities_mass: list[int]
    """
        Вместимость машины по объему. (объем должен быть целым числом  V_car_int = int(VOLUME_SCALE * V_car))
    """
    vehicle_capacities_volume: list[float]
    """
        Номер депо, можно считать всегда 0. 
    """
    depo: int


class SolutionCallback:

    def __init__(self, routing: pywrapcp.RoutingModel):
        self.model = routing
        self._best_objective = 1e10
        self.lock = Lock()

    def __call__(self):
        with self.lock:
            value = self.model.CostVar().Max()
            self._best_objective = min(self._best_objective, value)
            best = self._best_objective
        logger.info(f'find new solution: {value}, best solution: {best}')


def get_optimal_model_params() -> pywrapcp.DefaultRoutingSearchParameters:
    """
    Оптимальные параметры для модели.
    При изменении модели (например новые ограничения), эти параметры могут стать не самыми лучшими и потребуется доп
    тюнинг.
    Менять параметры стоит для эксперимента или при необходимости.
    :return: Оптимальные параметры для модели
    """
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.guided_local_search_lambda_coefficient = 0.0725
    search_parameters.local_search_operators.use_exchange = BOOL_TRUE
    search_parameters.local_search_operators.use_cross = BOOL_TRUE
    search_parameters.local_search_operators.use_tsp_opt = BOOL_TRUE
    #
    # # Disable operators that don't work well with PDP constraints
    search_parameters.local_search_operators.use_relocate = BOOL_FALSE  # Can break pickup-delivery pairs
    search_parameters.local_search_operators.use_or_opt = BOOL_FALSE  # Often ineffective for PDP

    search_parameters.local_search_operators.use_shortest_path_swap_active = "BOOL_FALSE"
    search_parameters.local_search_operators.use_relocate_neighbors = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_cross_exchange = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_relocate_and_make_active = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_extended_swap_active = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_shortest_path_swap_active = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_relocate_pair = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_light_relocate_pair = 'BOOL_FALSE'

    search_parameters.use_cp_sat = True
    search_parameters.use_generalized_cp_sat = True
    search_parameters.sat_parameters.num_search_workers = 16
    return search_parameters


def create_data_model(problem_data: ProblemData) -> ModelData:
    """ Создает данные для работы модели.
    :param problem_data: Основные данные о задаче
    :return: данные для работы модели
    """
    logger.info("prepare data for model")

    model_data = ModelData()
    model_data.problem_data = problem_data

    """
    Тут небольшой костыль (но вполне законный).
    Так как обычно задачи маршрутизации решаются с использованием депо, а депо в текущей задачу нет, то тут добавляется 
    "призрачное" депо, чтобы модель могла свободно отработать.
    Для данного депо (0 точка) все расстояния до других точек равны 0.  
    """
    N = problem_data.N
    arr = np.zeros((N + 1, N + 1), dtype=np.int64)
    arr[1:, 1:] = nx.adjacency_matrix(problem_data.graph, weight='length').toarray()
    model_data.dsts = arr

    model_data.num_vehicles = len(problem_data.cars)
    # Формирование списка заказов. В нем содержатся списки вида (начальная нода, вторая нода, ... , конечная нода)
    pickups_deliveries = []
    for u, d in problem_data.graph.nodes(data=True):
        if d['start']:
            pd = [problem_data.node2index[u]]
            v = d['next_node']
            while problem_data.graph.nodes()[v]['next_node'] is not None:
                pd.append(problem_data.node2index[v])
                v = problem_data.graph.nodes()[v]['next_node']
                if u == v:
                    raise CargoLoopException()
            pd.append(problem_data.node2index[v])
            pickups_deliveries.append(pd)
    model_data.pickups_deliveries = pickups_deliveries
    # установление спроса по объему и массе
    model_data.demands_mass = [0] + [int(m) for _, m in problem_data.graph.nodes(data='mass')]
    model_data.demands_volume = [0] + [int(m * ModelData.VOLUME_SCALE) for _, m in
                                       problem_data.graph.nodes(data='volume')]

    # установление вместимости по объему и массе (учитывая коэфф загруженности)
    vehicle_capacities_mass = []
    vehicle_capacities_volume = []
    for car in problem_data.cars:
        vehicle_capacities_mass.append(int(car.max_mass))
        vehicle_capacities_volume.append(int(car.max_volume))

    model_data.vehicle_capacities_mass = vehicle_capacities_mass
    model_data.vehicle_capacities_volume = vehicle_capacities_volume

    # номер фиктивного депо
    model_data.depo = 0
    return model_data


def get_solution(model_data: ModelData, manager, routing, solution) -> tuple[float, list[list[int]]]:
    """
    Восстановление маршрутов из найденного решения.

    :param model_data: Данные модели
    :param manager: менеджер решения
    :param routing: модель
    :param solution: найденное решение
    :return: скор и лист путей (пути в номерах, исключая депо (точку 0), а не в исходных нодах)
    """
    total = 0
    result = []
    index2node = model_data.problem_data.index2node
    for vehicle_id in range(model_data.num_vehicles):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            result.append([])
            continue
        index = routing.Start(vehicle_id)
        path = []
        route_cost = 0
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        path.append(manager.IndexToNode(index))
        if len(path) == 2 and path[0] == path[1] == 0:
            path = []
        total += route_cost
        result.append([index2node[p] for p in path if p != 0])

    return total, result


def add_mass_constraint(
        model_data: ModelData,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager
):
    """
    Добавление ограничение на вместимость по массе
    :param model_data: данные модели
    :param routing: солвер модели
    :param manager: менеджер модели
    :return: None
    """

    def demand_mass_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return model_data.demands_mass[from_node]

    logger.info('Добавление ограничений для массы')
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_mass_callback),
        0,  # null capacity slack
        model_data.vehicle_capacities_mass,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity_mass"
    )


def add_volume_constraint(
        model_data: ModelData,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager
):
    """
    Добавление ограничение на вместимость по объему
    :param model_data: данные модели
    :param routing: солвер модели
    :param manager: менеджер модели
    :return: None
    """

    def demand_volume_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return model_data.demands_volume[from_node]

    logger.info('Добавление ограничений для объема')
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_volume_callback),
        0,  # null capacity slack
        model_data.vehicle_capacities_volume,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity_volume"
    )


def add_time_window(
        model_data: ModelData,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        time_dimension_name: str = 'time'
):
    problem_data: ProblemData = model_data.problem_data
    graph: nx.Graph = problem_data.graph

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == 0:
            return 0
        from_node = problem_data.index2node[from_node]
        if to_node == 0:
            return int(graph.nodes()[from_node]['service_time'])
        to_node = problem_data.index2node[to_node]

        return int(graph.edges()[from_node, to_node]['time'] + graph.nodes()[from_node]['service_time'])

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    routing.AddDimension(
        transit_callback_index,
        0,  # allow waiting time
        CAR_TIME_UPPER_BOUND,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time_dimension_name,
    )
    
    # time_dimension = routing.GetDimensionOrDie(time_dimension_name)

    # min_time = min(d for _, d in graph.nodes(data='start_time'))

    # for u, d in graph.nodes(data=True):
    #     index = manager.NodeToIndex(problem_data.node2index[u])

    #     from_time = int((d['start_time'] - min_time).total_seconds() // 60)
    #     to_time = int((d['end_time'] - min_time).total_seconds() // 60)
    #     # print(from_time, to_time)
    #     time_dimension.CumulVar(index).SetRange(from_time, to_time)


def add_pick_up_and_delivery(
        model_data: ModelData,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        count_dimension_name: str = 'count'):
    """
    Добавление ограничения на порядок посещения.
    :param count_dimension_name: Название размерности для отслеживания порядка
    :param model_data: данные модели
    :param routing: солвер модели
    :param manager: менеджер модели
    """
    # Define Transportation Requests.
    logger.info('Добавление ограничения для порядка доставки')
    count_dimension = routing.GetDimensionOrDie(count_dimension_name)
    for request in model_data.pickups_deliveries:
        for i in range(len(request) - 1):
            pickup_index = manager.NodeToIndex(request[i])
            delivery_index = manager.NodeToIndex(request[i + 1])

            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            # поднять и сбросить груз должна одна машина.
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            # номер ноды c "поднятием" груза раньше ноды со "сбрасыванием" груза
            routing.solver().Add(
                count_dimension.CumulVar(pickup_index) <=
                count_dimension.CumulVar(delivery_index)
            )


def add_distance_constraint(
        model_data: ModelData,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        distance_dimension_name: str = 'distance'
):
    """
    Добавление ограничения на пройденное расстояние для машины.
    :param distance_dimension_name: Название размерности для отслеживания расстояния.
    :param model_data: Данные модели
    :param routing: Солвер модели
    :param manager: Менеджер модели
    """
    # count_dimension.Set
    solver = routing.solver()
    distance_dimension = routing.GetDimensionOrDie(distance_dimension_name)
    logger.info('Добавление ограничений для пройденного расстояния')
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        # Vehicle is used if next node after start is not the end
        is_used = routing.VehicleRouteConsideredVar(vehicle_id)
        # Get the cumulative distance at the end of the route
        end_cumul_var = distance_dimension.CumulVar(routing.End(vehicle_id))
        # Enforce minimal distance if used

        car = model_data.problem_data.cars[vehicle_id]
        min_dst = car.tariff_cost.min_dst_km
        max_dst = car.tariff_cost.max_dst_km

        solver.Add(
            end_cumul_var + 10 >= int(min_dst * 1000) - CAR_DISTANCE_UPPER_BOUND * 100 * (1 - is_used)
        )
        solver.Add(
            end_cumul_var - 10 <= int(max_dst * 1000)
        )


def add_vehicles_cost(model_data: ModelData,
                      routing: pywrapcp.RoutingModel,
                      manager: pywrapcp.RoutingIndexManager):
    """
    Добавление обработчиков на стоимость машины.
    :param model_data: Данные модели
    :param routing: Солвер модели
    :param manager: Менеджер модели
    """
    logger.info('Добавление стоимостей машин')
    problem_data = model_data.problem_data

    def cost_callback(from_index, to_index, cost_per_km):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return model_data.dsts[from_node][to_node] * cost_per_km

    for vehicle_id in range(manager.GetNumberOfVehicles()):
        car = problem_data.cars[vehicle_id]
        if car.tariff_cost.cost_per_km == 0:
            routing.SetFixedCostOfVehicle(int(car.tariff_cost.fixed_cost * 1000), vehicle_id)
        else:
            tc = routing.RegisterTransitCallback(
                partial(cost_callback, cost_per_km=int(car.tariff_cost.cost_per_km))
            )
            routing.SetArcCostEvaluatorOfVehicle(tc, vehicle_id)


def add_distance_dimension(model_data: ModelData,
                           routing: pywrapcp.RoutingModel,
                           manager: pywrapcp.RoutingIndexManager,
                           distance_dimension_name: str = 'distance'):
    """
    Добавление размерности для пройденного расстояния.
    :param distance_dimension_name: Имя размерности
    :param model_data: Данные модели
    :param routing: Солвер модели
    :param manager: Менеджер модели
    :return: distance_dimension
    """
    logger.info('Добавление размерности для расстояния')

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return model_data.dsts[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        CAR_DISTANCE_UPPER_BOUND,  # vehicle maximum travel distance
        True,  # start cumul to zero
        distance_dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(distance_dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(0)
    return distance_dimension


def add_count_dimension(routing: pywrapcp.RoutingModel,
                        count_dimension_name: str = 'count'):
    """
    Добавление размерности для пройденного расстояния.
    :param count_dimension_name: Имя размерности для подсчета кол-ва посещенных точек
    :param routing: Солвер модели
    :return count_dimension
    """
    logger.info('Добавление размерности для расстояния')

    def count_callback(*args):
        return 1

    count_callback_index = routing.RegisterTransitCallback(count_callback)

    routing.AddDimension(
        count_callback_index,
        0,  # no slack
        MAX_POINT_NUMBER,  # vehicle maximum travel distance
        True,  # start cumul to zero
        count_dimension_name,
    )
    count_dimension = routing.GetDimensionOrDie(count_dimension_name)
    count_dimension.SetGlobalSpanCostCoefficient(0)
    return count_dimension


def do_solve(
        problem_data: ProblemData,
        *,
        time=DEFAULT_MINUTES_FOR_MODEL,
        solution_limit=None,
        search_parameters: Optional[pywrapcp.DefaultRoutingSearchParameters] = None,
        initial_solution=None) -> list[list[object]]:
    """
        Описание основной проблемы.

    :param problem_data: Описание проблемы
    :param time: Ограничение по времени, по умолчанию DEFAULT_MINUTES_FOR_MODEL.
    :param solution_limit: Ограничение на кол-во найденных решений.
    :param search_parameters:  Параметры поиска, по умолчанию берутся из get_optimal_model_params()
    :param initial_solution:  Начальное решение, от которого будет запускаться поиск
    (если не указано, то солвер сам найдет решение).
    :return:  Либо картеж (скор, список путей, где путь это лист индексов посещенных нод) если решение найдено,
     либо None если не найдено.
    """
    # создание данных для модели
    model_data: ModelData = create_data_model(problem_data)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(model_data.problem_data.graph.nodes()) + 1, model_data.num_vehicles, model_data.depo
    )
    logger.info("Начало создания модели")
    routing = pywrapcp.RoutingModel(manager)

    for u in model_data.problem_data.graph.nodes():
        for v in model_data.problem_data.graph.nodes():
            if u != v and (u, v) not in model_data.problem_data.graph.edges():
                i_index = manager.NodeToIndex(problem_data.node2index[u])
                j_index = manager.NodeToIndex(problem_data.node2index[v])
                # если по каким-либо эвристическим соображениям часть ребер можно убрать, то их можно не рассматривать
                routing.NextVar(i_index).RemoveValue(j_index)

    add_distance_dimension(model_data, routing, manager)
    add_count_dimension(routing)

    add_vehicles_cost(model_data, routing, manager)
    add_distance_constraint(model_data, routing, manager)
    add_pick_up_and_delivery(model_data, routing, manager)
    add_mass_constraint(model_data, routing, manager)
    add_volume_constraint(model_data, routing, manager)
    add_time_window(model_data, routing, manager)

    if not search_parameters:
        search_parameters = get_optimal_model_params()
        search_parameters.time_limit.seconds = int(60 * time)
        if solution_limit is not None:
            search_parameters.solution_limit = solution_limit

    search_parameters.log_search = False

    routing.AddAtSolutionCallback(SolutionCallback(routing))

    routing.CloseModelWithParameters(search_parameters)

    logger.info(f'Начало решения')
    if initial_solution:
        logging.info(f'Use initial_solutions: {len(initial_solution)}')
        sols = []
        for s in initial_solution:
            if len(s) > 0 and s[0] == s[-1] == 0:
                s = s[1:-1]
            sols.append(s)
        assignment = routing.ReadAssignmentFromRoutes(sols, True)

        if not assignment:
            logging.warning(f'Bad Initial Solutions: {initial_solution}')
        solution = routing.SolveFromAssignmentWithParameters(
            assignment, search_parameters
        )

    else:
        solution = routing.SolveWithParameters(search_parameters)
    if solution:
        logger.info(f'find solution')
        score, solution = get_solution(model_data, manager, routing, solution)
        logger.info(f"best_score: {score / 1000:.2f}")
        return solution
    else:
        logger.warning("No solution found !")
        return []


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_cars(
        tariffs: list[Tariff],
        repeat_num=25
) -> list[Car]:
    repeated_cars = []
    for tariff in tariffs:
        for tc in tariff.cost_per_distance:
            for i in range(repeat_num):
                # дублирование автопарка, чтобы использовать все имеющиеся машины
                repeated_cars.append(
                    create_car(len(repeated_cars), tariff, tc)
                )
    return repeated_cars


def create_car(car_id: int, tariff: Tariff, tc: TariffCost):
    return Car(
        id=car_id,
        tariff_id=tariff.id,
        max_mass=int(tariff.mass * Tariff.DEFAULT_MASS_UTILIZATION),
        max_volume=int(tariff.volume * Tariff.DEFAULT_VOLUME_UTILIZATION * ModelData.VOLUME_SCALE),
        tariff_cost=tc
    )


def get_data(
        g: nx.Graph,
        cars: list[Car]
) -> ProblemData:
    data = ProblemData(
        graph=g,
        N=len(g.nodes()),
        node2index={u: i + 1 for i, u in enumerate(g.nodes())},
        index2node={i + 1: u for i, u in enumerate(g.nodes())},
        cars=cars
    )
    return data


def solve_sub_problem(g: nx.Graph, sub_nodes: list[object], tariffs: list[Tariff]):
    logger.info("start solve sub problem with part {}".format(len(sub_nodes)))
    # обход в глубину для выявление всех нод заказов
    nodes = []
    for u in sub_nodes:
        nodes.append(u)
        v = g.nodes()[u]['next_node']
        while g.nodes()[v]['next_node'] is not None:
            nodes.append(v)
            v = g.nodes()[u]['next_node']
        nodes.append(v)
    sub_g = g.subgraph(nodes)
    sub_cars = get_cars(tariffs, len(sub_g.nodes()) // 2)
    sub_data = get_data(sub_g, sub_cars)
    # решение подзадачи
    sol = do_solve(sub_data, time=1, solution_limit=20)
    for i, path in enumerate(sol):
        if len(path) > 0:
            yield sub_cars[i], path


def get_initial_sols_from_sub_solutions(
        sub_sols: list[tuple[Car, list]],
        tariffs: list[Tariff]
) -> list[tuple[Car, list]]:
    def car_key(k: tuple[Car, list]):
        car = k[0]
        return car.tariff_id, car.tariff_cost

    cars = {k: list(path for (car, path) in v) for k, v in groupby(sorted(sub_sols, key=car_key), key=car_key)}
    init_sols = []
    for tariff in tariffs:
        for tc in tariff.cost_per_distance:
            for i in range(5):
                init_sols.append((create_car(len(init_sols), tariff, tc), []))
            for sol in cars.get((tariff.id, tc), []):
                init_sols.append((create_car(len(init_sols), tariff, tc), sol))

    return init_sols


def convert_to_routes(
        data: ProblemData,
        solution: list[list],
        tariffs: list[Tariff]
) -> list[Route]:
    all_routes = []
    tariff_id2tariff = {t.id: t for t in tariffs}

    for i, path in enumerate(solution):
        if len(path) > 0:
            all_routes.append(
                Route(
                    id=i,
                    path=path,
                    tariff=tariff_id2tariff[data.cars[i].tariff_id]
                ))
    return all_routes


def solve_with_batch(
        g: nx.Graph,
        tariffs: list[Tariff],
        *,
        max_problem_size: int = 50
) -> list[Route]:
    # если задача мала, то решаем ее сразу
    if len(g.nodes()) < max_problem_size:
        repeated_cars = get_cars(tariffs, len(g.nodes()) // 2)
        data = get_data(g, repeated_cars)
        return convert_to_routes(data, do_solve(data), tariffs)
    # для большой задачи бьем ее на части, чтобы найти начальное решение
    start_nodes = [u for u, d in g.nodes(data=True) if d['start']]
    random.shuffle(start_nodes)
    parts = len(start_nodes) // max_problem_size + 1

    sub_solutions = []
    for sn in split(start_nodes, parts):
        for (car, path) in solve_sub_problem(g, sn, tariffs):
            sub_solutions.append((car, path))
    # собираем решение для частей, чтобы использовать их как начальное решение для общей задачи
    init_sols = get_initial_sols_from_sub_solutions(sub_solutions, tariffs)
    cars = [i[0] for i in init_sols]
    data = get_data(g, cars)
    init_sols = [[data.node2index[p] for p in i[1]] for i in init_sols]
    return convert_to_routes(data, do_solve(data, initial_solution=init_sols), tariffs)


def find_optimal_paths(
        g: nx.Graph,
        tariffs: list[Tariff]
) -> list[Route]:
    """
    Входной метод для построения маршрутов. Принимает на вход граф заявок и список доступных машин.
    Преобразует данные в данные для работы модели, вызывает модель и парсит результат.

    :param g: Граф Заявок
    :param tariffs: Список тарифов, которые могут развозить груз
    :return: list[Route] - список путей, содержит информация о пути (лист точек в графе заказов)
     и машины для каждого маршрута
    """
    logger.info(f'problem size: {len(g.nodes())}')

    return solve_with_batch(g, tariffs)
