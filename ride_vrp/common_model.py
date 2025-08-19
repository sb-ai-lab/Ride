from typing import TypeVar, Protocol, runtime_checkable

from vrp_study.routing_manager import RoutingManager, InnerNode

M_cov = TypeVar('M_cov', covariant=True)
M_contr = TypeVar('M_contr', contravariant=True)


@runtime_checkable
class ModelFactory(Protocol[M_cov]):
    def build_model(self, routing_manager: RoutingManager) -> M_cov:
        pass


@runtime_checkable
class Solver(Protocol[M_contr]):
    def solve(self, model: M_contr) -> list[list[InnerNode]] | None:
        pass


@runtime_checkable
class SolverFactory(Protocol[M_contr]):
    def build_solver(self, model: M_contr) -> Solver[M_contr]:
        pass
