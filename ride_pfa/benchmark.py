import logging
from dataclasses import dataclass
from multiprocessing import Pool
from time import sleep

import numpy as np

from .path_finding.pfa import PathFinding
from .tqdm_progress_bar import with_progress
from .utils import get_execution_time

__all__ = [
    "Statistics",
    "PfaComparator"
]

log = logging.getLogger(__name__)


class Statistics:
    def __init__(self):
        self.baseline_length: list[float] = []
        self.baseline_path: list[list[int]] = []
        self.baseline_time: list[float] = []

        self.test_length: list[float] = []
        self.test_path: list[list[int]] = []
        self.test_time: list[float] = []

    def __add__(self, other):
        if not isinstance(other, Statistics):
            raise Exception('what do you do? It should be a Statistics')
        s = Statistics()
        for f in vars(s):
            if not f.startswith('__'):
                s.__dict__[f] = self.__dict__[f] + other.__dict__[f]
        return s

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self + other

    def get_err(self) -> np.ndarray:
        a = np.array(self.baseline_length)
        b = np.array(self.test_length)
        delta = (b - a) / a * 100
        return delta

    def get_acceleration(self) -> np.ndarray:
        a = np.array(self.baseline_time)
        b = np.array(self.test_time)
        acceleration = a / b
        return acceleration

    def __repr__(self) -> str:
        e, a = self.get_err(), self.get_acceleration()
        return f"""
        err_max:    {e.max():.4f}
        err_min:    {e.min():.4f}
        err_median: {np.median(e):.4f}
        err_mean:   {e.mean():.4f}

        acceleration_max:    {a.max():.4f}
        acceleration_min:    {a.min():.4f}
        acceleration_median: {np.median(a):.4f}
        acceleration_mean:   {a.mean():.4f}
        """


@dataclass
class PfaComparator:
    baseline: PathFinding
    test_algorithm: PathFinding
    points: list[tuple[int, int]]

    workers: int = 4
    iterations: int = 4
    with_tqdm_log: bool = True

    def test(self, pfa: PathFinding, u: int, v: int):
        def func():
            return pfa.find_path(u, v)

        return get_execution_time(func, iterations=self.iterations)

    def do_calc(self, data_partitions):
        point_partition, worker_number = data_partitions

        stat = Statistics()

        log.debug('start %i workers', worker_number)

        if self.with_tqdm_log:
            # For tqdm loading in notebooks
            sleep(worker_number / 10)
            _iter = with_progress(
                point_partition,
                desc='find paths',
                position=worker_number
            )
        else:
            _iter = point_partition

        for p1, p2 in _iter:
            time_l, (l, p) = self.test(self.baseline, p1, p2)

            time_h, (h_l, h_p) = self.test(self.test_algorithm, p1, p2)

            stat.baseline_time.append(time_l)
            stat.baseline_length.append(l)
            stat.baseline_path.append(p)

            stat.test_time.append(time_h)
            stat.test_length.append(h_l)
            stat.test_path.append(h_p)

        return stat

    def compare(self) -> Statistics:
        data = [([p for p in self.points[i::self.workers]], i) for i in range(self.workers)]
        if self.workers == 1:
            stat = self.do_calc(data[0])
        else:
            with Pool(self.workers) as p:
                stat = sum(p.imap_unordered(self.do_calc, data))
        return stat
