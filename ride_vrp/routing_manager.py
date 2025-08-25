from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Self

import numpy as np

from .data_model import Tariff, Cargo, Node
from .data_model import TariffCost
from .distance_matrix import DistanceMatrix


@dataclass
class InnerNode:
    id: int
    start_time: int
    end_time: int
    service_time: int
    demand: np.ndarray
    is_transit: bool = field()
    pdp_id: int = field(default=-1)
    routing_node: Node | None = field(default=None)


@dataclass
class InnerCar:
    id: int = field(hash=True)
    tariff: Tariff = field(hash=False)
    capacity: np.ndarray = field(hash=False)
    tariff_cost: TariffCost = field()

    end_node: InnerNode
    start_node: InnerNode
    use_when_empty: bool


@dataclass
class Pdp:
    id: int = field(hash=True)
    nodes: list[InnerNode] = field()


@dataclass
class RoutingManager:
    _dsts: np.ndarray
    _time: np.ndarray
    _inner_nodes: list[InnerNode]
    _inner_cars: list[InnerCar]
    _pick_up_and_delivery_nodes: list[Pdp]
    _depo_index: int
    _id2index: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        self._id2index = {n.id: i for i, n in enumerate(self._inner_nodes)}

    def get_node(self, node_id: int) -> InnerNode:
        return self.nodes()[self._id2index[node_id]]

    def get_index(self, node: InnerNode):
        return self._id2index[node.id]

    def get_distance(self, node_a: InnerNode, node_b: InnerNode) -> float:
        return float(self._dsts[node_a.id, node_b.id])

    def get_time(self, node_a: InnerNode, node_b: InnerNode) -> float:
        return float(self._time[node_a.id, node_b.id])

    def starts_ids(self):
        return [car.start_node.id for car in self._inner_cars]

    def ends_ids(self):
        return [car.end_node.id for car in self._inner_cars]

    def cars(self) -> list[InnerCar]:
        return self._inner_cars

    def nodes(self) -> list[InnerNode]:
        return self._inner_nodes

    def get_pick_up_and_delivery_nodes(self) -> list[list[int]]:
        return [[self._id2index[n.id] for n in pdp.nodes] for pdp in self._pick_up_and_delivery_nodes]

    def get_depo_index(self) -> int:
        return self._depo_index

    def sub_problem(self, nodes: list[InnerNode], cars: list[InnerCar]):
        assert len(cars) > 0
        pdp_ids = {n.pdp_id for n in nodes}
        nodes_ids = {n.id for n in nodes}
        if self._depo_index not in nodes_ids:
            nodes = [self._inner_nodes[self._depo_index]] + nodes
        res = RoutingManager(
            _dsts=self._dsts,
            _time=self._time,
            _inner_nodes=nodes,
            _inner_cars=cars,
            _pick_up_and_delivery_nodes=[pdp for pdp in self._pick_up_and_delivery_nodes if pdp.id in pdp_ids],
            _depo_index=self._depo_index
        )
        return res


class RoutingManagerBuilder(ABC):
    def __init__(self,
                 distance_matrix: DistanceMatrix
                 ):

        self.distance_matrix: DistanceMatrix = distance_matrix

        self._nodes: list[Node] = []
        self._depots: list[Node] = []
        self._cargos: list[Cargo] = []
        self._tariffs: list[Tariff] = []

        self._np_dsts: np.ndarray
        self._np_time: np.ndarray

        self._inner_nodes: list[InnerNode] = []
        self._inner_cars: list[InnerCar] = []

        self._pdp: list[Pdp] = []

    @property
    def cargos(self) -> list[Cargo]:
        return self._cargos

    @property
    def tariffs(self) -> list[Tariff]:
        return self._tariffs

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def depots(self) -> list[Node]:
        return self._depots

    def add_tariff(self, tariff: Tariff):
        self._tariffs.append(tariff)

    def add_tariffs(self, tariffs: Iterable[Tariff]):
        for t in tariffs:
            self.add_tariff(t)

    def with_tariffs(self, tariffs: Iterable[Tariff] | Tariff) -> Self:
        if isinstance(tariffs, Iterable):
            self.add_tariffs(tariffs)
        else:
            self.add_tariff(tariffs)
        return self

    def add_node(self, node: Node):
        self._nodes.append(node)

    def add_nodes(self, nodes: Iterable[Node]):
        for n in nodes:
            self.add_node(n)

    def with_nodes(self, nodes: Iterable[Node]) -> Self:
        self.add_nodes(nodes)
        return self

    def add_cargo(self, cargo: Cargo):
        self._cargos.append(cargo)
        for n in cargo.nodes:
            self.add_node(n)

    def add_cargos(self, cargos: Iterable[Cargo]):
        for c in cargos:
            self.add_cargo(c)

    def with_cargos(self, cargos: Iterable[Cargo]) -> Self:
        self.add_cargos(cargos)
        return self

    def add_depot(self, depo: Node):
        self._depots.append(depo)

    def add_depots(self, depots: Iterable[Node]):
        for depo in depots:
            self.add_depot(depo)

    def with_depo(self, depots: Iterable[Node] | Node) -> Self:
        if isinstance(depots, Iterable):
            self.add_depots(depots)
        else:
            self.add_depot(depots)
        return self

    def build(self) -> RoutingManager:
        self._validate()
        return self._build()

    def _validate(self):
        # todo
        ...

    def _build(self) -> RoutingManager:

        self._create_inner_nodes()

        self._node_to_inner_node = {n.routing_node: n for n in self._inner_nodes if n.routing_node is not None}

        self._create_inner_cars()

        self._sort_nodes()

        self._build_distance_matrix()

        return RoutingManager(
            _dsts=self._np_dsts,
            _time=self._np_time,
            _inner_nodes=self._inner_nodes,
            _inner_cars=self._inner_cars,
            _pick_up_and_delivery_nodes=self._pdp,
            _depo_index=0
        )

    @abstractmethod
    def _create_inner_nodes(self):
        ...

    @abstractmethod
    def _create_inner_cars(self):
        ...

    def _sort_nodes(self):
        starts = set(car.start_node.id for car in self._inner_cars)
        ends = set(car.end_node.id for car in self._inner_cars)

        def key(node: InnerNode):
            if node.id in starts:
                return -1
            if node.id in ends:
                return 1
            return 0

        self._inner_nodes.sort(key=key)
        for i, n in enumerate(self._inner_nodes):
            n.id = i

    def _build_distance_matrix(self):
        num_nodes = len(self._inner_nodes)
        dsts = np.zeros((num_nodes, num_nodes))
        time = np.zeros((num_nodes, num_nodes))

        for i, n1 in enumerate(self._inner_nodes):
            for j, n2 in enumerate(self._inner_nodes):
                if n1.routing_node is None or n2.routing_node is None or i == j:
                    continue
                if n1.routing_node.id == n2.routing_node.id:
                    continue
                dsts[i, j] = self.distance_matrix.get_distance(n1.routing_node.id, n2.routing_node.id)
                time[i, j] = self.distance_matrix.get_time(n1.routing_node.id, n2.routing_node.id)
        self._np_dsts = dsts
        self._np_time = time
