import logging
import time
from threading import Lock

from ortools.constraint_solver import pywrapcp

log = logging.getLogger(__name__)


class SolutionCallback:

    def __init__(self):
        self._routing: pywrapcp.RoutingModel | None = None

    def solve_with(self, routing: pywrapcp.RoutingModel):
        self._routing = routing

    def end_solve(self):
        self._routing = None

    def __call__(self):
        pass


class LoggingCallback(SolutionCallback):

    def __init__(self):
        super().__init__()
        self._best_objective = 1e10
        self.count = 0
        self.lock = Lock()
        self.start_time = time.time()

    def solve_with(self, routing: pywrapcp.RoutingModel):
        super().solve_with(routing)
        self.start_time = time.time()

    def __call__(self):
        if not self._routing:
            return
        self.count += 1
        count = self.count
        value = self._routing.CostVar().Max()
        self._best_objective = min(self._best_objective, value)
        best = self._best_objective
        delta = time.time() - self.start_time
        log.info(f'time: {delta:.3f}; new solution ({count}): {value}; best solution: {best}')
