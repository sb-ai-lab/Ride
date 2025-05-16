from dataclasses import dataclass, field
from typing import List

"""
Описание основных моделей данных для модели
"""

__all__ = [
    'Cargo',
    'MergedCargo',
    'Route',
    'Tariff',
    'TariffCost'
]


@dataclass(
    slots=True
)
class Cargo:
    """
    Класс, который описывает груз
    """
    """
    Идентификатор груза
    """
    id: object = field(hash=True)
    """
    Масса груза в разрезе посещаемых точек (положительна, если загрузка, отрицательна для разгрузки)
    """
    mass: List[float] = field(hash=False)
    """
    Объем груза в разрезе посещаемых точек (положительный, если загрузка, отрицательный для разгрузки)
    """
    volume: List[float] = field(hash=False)
    """
    Список всех адресов (включая начальную и конечную)
    """
    nodes: List[object] = field(hash=False)
    """
    Время на обслуживание каждой точки
    """
    service_time_minutes: List[int]


@dataclass(
    frozen=True,
    order=True
)
class TariffCost:
    # минимальное расстояние по текущему тарифу
    min_dst_km: float = field()
    # максимальное расстояние по текущему тарифу
    max_dst_km: float = field()
    # цена за километр в рублях
    cost_per_km: float = field()
    # фиксированная цена в рублях
    fixed_cost: float = field()


@dataclass(
    frozen=True
)
class Tariff:
    DEFAULT_VOLUME_UTILIZATION = 0.75  # на сколько машина может быть утилизирована по объему
    DEFAULT_MASS_UTILIZATION = 0.75  # на сколько машина может быть утилизирована по массе

    # идентификатор тарифа (например "эконом")
    id: object = field()
    mass: float = field(hash=False)  # вместимость по массе
    volume: float = field(hash=False)  # вместимость по объему

    """
        список "сегментов" - цен на разном километраже.
        Например, если тариф до 5км фиксированный (100р.), а после 5 км тариф линейный (100 * км + 500), то в Tariff
        задается так:
        Tariff(
                id='ТС750',
                mass=750,
                volume=10, 
                cost_per_distance=[
                    TariffCost(min_dst_km=0, max_dst_km=5, cost_per_km=0, fixed_cost=100),
                    TariffCost(min_dst_km=5, max_dst_km=float('inf'), cost_per_km=100, fixed_cost=500)
                ]
            )
    """
    cost_per_distance: List[TariffCost] = field(hash=False)


@dataclass(
    slots=True
)
class MergedCargo(Cargo):
    """
        Класс составного груза, хранит все id грузов, из которых он состоит
    """
    merged_ids: list[object] = field(default_factory=list, repr=True, hash=False)


@dataclass
class Route:
    """
        Класс маршрута
    """
    id: int = field(hash=True)  # идентиификатор маршрута
    path: list[object] = field(hash=False)  # путь
    tariff: Tariff = field(hash=False)  # тип машины на данном маршруту
