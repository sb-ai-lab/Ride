import logging

from scripts.data_model import Cargo

__all__ = [
    'greedy_merge_cargo',
]

from scripts.exceptions import TooBigCargoException

logger = logging.getLogger('knapsack')


def greedy_merge_cargo(
        cargos: list[Cargo],
        max_mass: float,
        max_volume: float
) -> dict[int, list[Cargo]]:
    """
    
    Цель метода объединить грузы так, чтобы для их перевозки было задействовано как можно меньше машин.
    Считается, что переданные на вход грузы имеют общее начало, конец и время и глобально могут рассматриваться как один
    груз.
    :raises TooBigCargoException: Возникает когда один или несколько из переданных грузов превышают максимально
    допустимую массу или объем
    
    :param cargos: список грузов для объединения
    :param max_mass: максимальная масса в "рюкзаке"
    :param max_volume: максимальный объем в "рюкзаке"
    :return: в какой рюкзак какие грузы в формате словаря {номер рюкзака: список грузов}
    """
    if len(cargos) <= 1:
        return {0: cargos}
    res = {}
    k = 0
    # список еще не размещенных грузов
    not_placed = [crg for crg in cargos]
    # количество грузов для отлова момента, когда грузы не влазят
    prev_len = len(not_placed)
    while len(not_placed) > 0:
        mass = 0
        volume = 0

        cargos = not_placed
        not_placed = []
        for crg in sorted(cargos, key=lambda c: (c.mass, c.volume)):
            if mass + crg.mass[1] > max_mass or volume + crg.volume[1] > max_volume:
                not_placed.append(crg)
            else:
                mass += crg.mass[1]
                volume += crg.volume[1]
                if k not in res:
                    res[k] = []
                res[k].append(crg)
        if len(not_placed) > 0:
            k += 1
        if prev_len == len(not_placed):
            raise TooBigCargoException(f'Слишком большие грузы: {not_placed}')
        prev_len = len(not_placed)
    return res
