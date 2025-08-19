from dataclasses import dataclass, field
from functools import partial
from typing import Self, Protocol, Iterable, Callable

from ortools.constraint_solver import pywrapcp
from ortools.util.optional_boolean_pb2 import BOOL_FALSE, BOOL_TRUE
from ride_vrp.common_model import ModelFactory, Solver
from ride_vrp.routing_manager import RoutingManager, InnerNode

from .callback import SolutionCallback
from ..configs import ModelConfig
from ..exceptions import SolutionNoFoundException
from ..initial_solution_builder import InitialSolutionBuilder


class VRPModel:
    def __init__(self, routing_manager: RoutingManager) -> None:
        self._index_manager = pywrapcp.RoutingIndexManager(
            len(routing_manager.nodes()),
            len(routing_manager.cars()),
            routing_manager.starts_ids(),
            routing_manager.ends_ids()
        )
        self._routing = pywrapcp.RoutingModel(self._index_manager)
        self._routing_manager = routing_manager

        self._constraints_names: set[str] = set()
        # self._dimensions_names: set[str] = set()
        # self._costs_names: set[str] = set()

    @property
    def index_manager(self) -> pywrapcp.RoutingIndexManager:
        return self._index_manager

    @property
    def routing(self) -> pywrapcp.RoutingModel:
        return self._routing

    @property
    def routing_manager(self) -> RoutingManager:
        return self._routing_manager

    @property
    def constraints_names(self):
        return self._constraints_names

    def add_constraint_name(self, constraint_name: str):
        self._constraints_names.add(constraint_name)


class OrToolsModelAugmentor(Protocol):
    def apply(self, model: VRPModel):
        pass


@dataclass
class DistanceDimensionAugmentor(OrToolsModelAugmentor):
    car_distance_upper_bound: int = field(default=int(1e6))
    name: str = field(default='distance')

    def apply(self, model: VRPModel):
        rm = model.routing_manager
        im = model.index_manager
        r = model.routing

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = rm.nodes()[im.IndexToNode(from_index)]
            to_node = rm.nodes()[im.IndexToNode(to_index)]
            return int(rm.get_distance(from_node, to_node))

        transit_callback_index = r.RegisterTransitCallback(distance_callback)

        r.AddDimension(
            transit_callback_index,
            0,  # no slack
            self.car_distance_upper_bound,  # vehicle maximum travel distance
            True,  # start cumul to zero
            self.name,
        )


@dataclass
class CapacityDimensionAugmentor(OrToolsModelAugmentor):
    name: str = field(default='capacity')
    index: int = field(default=0)

    def apply(self, model: VRPModel):
        routing_manager = model.routing_manager
        manager = model.index_manager
        routing = model.routing
        index = self.index

        def demand_mass_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = routing_manager.nodes()[manager.IndexToNode(from_index)]
            return from_node.demand[index]

        routing.AddDimensionWithVehicleCapacity(
            routing.RegisterUnaryTransitCallback(demand_mass_callback),
            0,  # null capacity slack
            [car.capacity[index] for car in routing_manager.cars()],  # vehicle maximum capacities
            True,  # start cumul to zero
            self.name
        )


@dataclass
class TimeDimensionAugmentor(OrToolsModelAugmentor):
    max_value: int = field()
    max_slack: int = field()
    time_dimension_name: str = field(default='time')

    def apply(self, model: VRPModel):
        routing_manager = model.routing_manager
        manager = model.index_manager
        routing = model.routing

        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = routing_manager.nodes()[manager.IndexToNode(from_index)]
            to_node = routing_manager.nodes()[manager.IndexToNode(to_index)]
            if from_node.id == to_node.id:
                return 0
            return int(routing_manager.get_time(from_node, to_node) + from_node.service_time)

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        routing.AddDimension(
            transit_callback_index,
            self.max_slack,  # allow waiting time
            self.max_value,  # maximum time per vehicle
            True,  # Don't force start cumul to zero.
            self.time_dimension_name,
        )


@dataclass
class TimeConstraintAugmentor(OrToolsModelAugmentor):
    time_dimension_name: str = field(default='time')

    def apply(self, model: VRPModel):
        routing = model.routing
        routing_manager = model.routing_manager
        manager = model.index_manager

        time_dimension = routing.GetDimensionOrDie(self.time_dimension_name)

        model.add_constraint_name('time window')

        for i, node in enumerate(routing_manager.nodes()):
            from_time = node.start_time
            to_time = node.end_time
            if node.is_transit:
                index = manager.NodeToIndex(i)
                time_dimension.CumulVar(index).SetRange(from_time, to_time)
            else:
                for j, car in enumerate(routing_manager.cars()):
                    if car.end_node.id == node.id:
                        index = routing.End(j)
                        time_dimension.CumulVar(index).SetRange(from_time, to_time)
                    elif car.start_node.id == node.id:
                        index = routing.Start(j)
                        time_dimension.CumulVar(index).SetRange(from_time, to_time)


@dataclass
class PDPConstraintAugmentor(OrToolsModelAugmentor):
    dimension_name: str = 'distance'

    def apply(self, model: VRPModel):
        routing = model.routing
        routing_manager = model.routing_manager
        manager = model.index_manager

        model.add_constraint_name('pdp')

        dimension = routing.GetDimensionOrDie(self.dimension_name)
        for request in routing_manager.get_pick_up_and_delivery_nodes():
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
                    dimension.CumulVar(pickup_index) <=
                    dimension.CumulVar(delivery_index)
                )


@dataclass
class LinearCostByDistanceAugmentor(OrToolsModelAugmentor):
    distance_dimension_name: str = 'distance'

    def apply(self, model: VRPModel):
        routing = model.routing
        routing_manager = model.routing_manager
        manager = model.index_manager

        def cost_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = routing_manager.nodes()[manager.IndexToNode(from_index)]
            to_node = routing_manager.nodes()[manager.IndexToNode(to_index)]
            return int(routing_manager.get_distance(from_node, to_node) * 100)

        routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost_callback))


class VRPModelFactory(ModelFactory[VRPModel]):

    def __init__(self):
        self._constraints: list[OrToolsModelAugmentor] = []
        self._dimensions: list[OrToolsModelAugmentor] = []
        self._costs: list[OrToolsModelAugmentor] = []

    def with_dimensions(self, dimensions: Iterable[OrToolsModelAugmentor] | OrToolsModelAugmentor,
                        *args: OrToolsModelAugmentor) -> Self:
        if isinstance(dimensions, Iterable):
            self._dimensions.extend(dimensions)
        else:
            self._dimensions.append(dimensions)
        self._dimensions.extend(args)
        return self

    def with_constraints(self, constraints: Iterable[OrToolsModelAugmentor] | OrToolsModelAugmentor,
                         *args: OrToolsModelAugmentor) -> Self:
        if isinstance(constraints, Iterable):
            self._constraints.extend(constraints)
        else:
            self._constraints.append(constraints)
        self._constraints.extend(args)
        return self

    def with_cost(self, costs: Iterable[OrToolsModelAugmentor] | OrToolsModelAugmentor,
                  *args: OrToolsModelAugmentor) -> Self:
        if isinstance(costs, Iterable):
            self._costs.extend(costs)
        else:
            self._costs.append(costs)
        self._costs.extend(args)
        return self

    def build_model(self, routing_manager: RoutingManager) -> VRPModel:
        vrp: VRPModel = VRPModel(routing_manager)
        for builders in [self._dimensions, self._constraints, self._costs]:
            for c in builders:
                c.apply(vrp)
        return vrp

    @classmethod
    def get_pdptw_model_factory(cls, routing_manager: RoutingManager) -> "VRPModelFactory":
        return VRPModelFactory().with_dimensions([
            DistanceDimensionAugmentor(),
            TimeDimensionAugmentor(
                max_slack=max(node.end_time for node in routing_manager.nodes()),
                max_value=max(node.end_time for node in routing_manager.nodes())
            ),
            CapacityDimensionAugmentor()
        ]).with_constraints([
            TimeConstraintAugmentor(),
            PDPConstraintAugmentor()
        ]).with_cost(
            LinearCostByDistanceAugmentor()
        )


def get_optimal_model_params_for_pdp(
        model_config: ModelConfig
) -> pywrapcp.DefaultRoutingSearchParameters:
    """
    Оптимальные параметры для модели.
    При изменении модели (например новые ограничения), эти параметры могут стать не самыми лучшими и потребуется доп
    тюнинг.
    Менять параметры стоит для эксперимента или при необходимости.
    :return: Оптимальные параметры для модели
    """
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = model_config.first_solution_type
    search_parameters.local_search_metaheuristic = model_config.ls_type
    search_parameters.guided_local_search_lambda_coefficient = 0.0725

    search_parameters.local_search_operators.use_relocate = BOOL_FALSE
    search_parameters.local_search_operators.use_relocate_pair = BOOL_TRUE
    search_parameters.local_search_operators.use_light_relocate_pair = BOOL_TRUE

    search_parameters.local_search_operators.use_relocate_neighbors = BOOL_FALSE
    search_parameters.local_search_operators.use_relocate_subtrip = BOOL_TRUE

    search_parameters.local_search_operators.use_exchange = BOOL_FALSE
    search_parameters.local_search_operators.use_exchange_pair = BOOL_TRUE
    search_parameters.local_search_operators.use_exchange_subtrip = BOOL_TRUE

    search_parameters.local_search_operators.use_relocate_expensive_chain = BOOL_FALSE
    search_parameters.local_search_operators.use_two_opt = BOOL_FALSE
    search_parameters.local_search_operators.use_or_opt = BOOL_FALSE
    search_parameters.local_search_operators.use_lin_kernighan = BOOL_FALSE
    search_parameters.local_search_operators.use_tsp_opt = BOOL_FALSE

    search_parameters.local_search_operators.use_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_relocate_and_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_exchange_and_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_exchange_path_start_ends_and_make_active = BOOL_FALSE
    search_parameters.local_search_operators.use_make_inactive = BOOL_FALSE
    search_parameters.local_search_operators.use_make_chain_inactive = BOOL_FALSE
    search_parameters.local_search_operators.use_swap_active = BOOL_FALSE

    search_parameters.local_search_operators.use_extended_swap_active = BOOL_FALSE
    search_parameters.local_search_operators.use_shortest_path_swap_active = BOOL_FALSE
    search_parameters.local_search_operators.use_shortest_path_two_opt = BOOL_FALSE
    search_parameters.local_search_operators.use_node_pair_swap_active = BOOL_FALSE

    # LNS
    search_parameters.local_search_operators.use_path_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_full_path_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_tsp_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_inactive_lns = BOOL_FALSE

    search_parameters.local_search_operators.use_global_cheapest_insertion_path_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_local_cheapest_insertion_path_lns = BOOL_TRUE

    search_parameters.local_search_operators.use_relocate_path_global_cheapest_insertion_insert_unperformed = BOOL_FALSE

    # вроде как не учитывает pd
    search_parameters.local_search_operators.use_global_cheapest_insertion_expensive_chain_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_local_cheapest_insertion_expensive_chain_lns = BOOL_FALSE

    # отменяет часть узлов (включая pick up and delivery) и заново вставляет используя указанные эвристики
    # global|local _cheapest_insertion
    search_parameters.local_search_operators.use_global_cheapest_insertion_close_nodes_lns = BOOL_FALSE
    search_parameters.local_search_operators.use_local_cheapest_insertion_close_nodes_lns = BOOL_TRUE
    search_parameters.heuristic_close_nodes_lns_num_nodes = 4

    search_parameters.use_cp_sat = True
    search_parameters.use_generalized_cp_sat = True
    search_parameters.sat_parameters.num_search_workers = 16

    search_parameters.time_limit.seconds = int(60 * model_config.max_execution_time_minutes)
    if model_config.max_solution_number > 0:
        search_parameters.solution_limit = model_config.max_solution_number

    return search_parameters


class VRPSolver(Solver[VRPModel]):

    def __init__(self,
                 initial_solution_builder: InitialSolutionBuilder | None = None,
                 model_config: ModelConfig | None = None,
                 callback: SolutionCallback | None = None,
                 parameters_provider: Callable[[], pywrapcp.DefaultRoutingSearchParameters] | None = None
                 ):
        self.initial_solution_builder = initial_solution_builder
        self.model_config = model_config or ModelConfig()
        self.callback = callback
        self.parameters_provider = parameters_provider or partial(get_optimal_model_params_for_pdp,
                                                                  model_config=self.model_config)

    def solve(self, model: VRPModel) -> list[list[InnerNode]] | None:
        if self.callback:
            self.callback.solve_with(model.routing)
            model.routing.AddAtSolutionCallback(self.callback)

        search_parameters = self.parameters_provider()
        search_parameters.log_search = False

        if self.initial_solution_builder:
            init_solutions = self.initial_solution_builder.get_initial_solution(
                model.routing_manager
            )

            indices_solutions: list[list[int]] = []
            for i, s in enumerate(init_solutions):
                if len(s) == 0:
                    indices_solutions.append([])
                else:
                    car = model.routing_manager.cars()[i]
                    start_end_set = {car.start_node.id, car.end_node.id}
                    path = [model.routing_manager.get_index(node) for node in s if node.id not in start_end_set]
                    indices_solutions.append(path)

            assignment = model.routing.ReadAssignmentFromRoutes(
                indices_solutions,
                True
            )

            final_solution = model.routing.SolveFromAssignmentWithParameters(
                assignment,
                search_parameters
            )
        else:
            final_solution = model.routing.SolveWithParameters(
                search_parameters
            )
        if self.callback:
            self.callback.end_solve()

        if final_solution:
            score, routes = get_solution(model, final_solution)
            return [[model.routing_manager.nodes()[i] for i in r] for r in routes]
        raise SolutionNoFoundException()


def get_solution(vrp_model: VRPModel, solution) -> tuple[float, list[list[int]]]:
    routing_manager, manager, routing = vrp_model.routing_manager, vrp_model.index_manager, vrp_model.routing
    total = 0
    result: list[list[int]] = []
    for vehicle_id, car in enumerate(routing_manager.cars()):
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
        route_cost += routing.GetFixedCostOfVehicle(vehicle_id)
        path.append(manager.IndexToNode(index))

        if len(path) == 2 and path[0] == path[1] == 0:
            path = []
        total += route_cost
        result.append(path)
    return total, result
