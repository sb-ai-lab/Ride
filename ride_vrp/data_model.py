import datetime
from dataclasses import dataclass, field
from typing import TypeVar, Generic

import numpy as np

__all__ = [
    'Node',
    'Cargo',
    'Tariff',
    'TariffCost',
    'Route'
]


@dataclass(
    frozen=True,
    slots=True
)
class Node:
    id: object
    cargo_id: object | None = field(hash=False, default=None)
    demand: np.ndarray | None = field(hash=False, default=None)
    start_time: datetime.datetime | int | None = field(hash=False, default=None)
    end_time: datetime.datetime | int | None = field(hash=False, default=None)
    service_time: datetime.timedelta | int | None = field(hash=True, default=None)
    coordinates: tuple[float, float] | None = field(hash=False, default=None)


@dataclass(
    frozen=True,
    slots=True
)
class Cargo:
    id: object = field(hash=True)
    nodes: list[Node] = field(hash=False)


@dataclass
class TariffCost:
    min_dst_km: float = field()
    max_dst_km: float = field()
    cost_per_km: float = field()
    fixed_cost: float = field()


@dataclass(
    frozen=True,
    slots=True
)
class Tariff:
    id: object = field()
    capacity: np.ndarray = field(hash=False)
    cost_per_distance: list[TariffCost] = field(hash=False)
    max_count: int = field(hash=False, default=-1)


T = TypeVar('T')


@dataclass
class Route(Generic[T]):
    id: int = field(hash=True)
    path: list[T] = field(hash=False)
    tariff: Tariff = field(hash=False)
