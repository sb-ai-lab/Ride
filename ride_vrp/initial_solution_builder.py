from abc import ABC, abstractmethod

from .routing_manager import InnerNode, RoutingManager


class InitialSolutionBuilder(ABC):
    @abstractmethod
    def get_initial_solution(self, routing_manager: RoutingManager) -> list[list[InnerNode]]:
        pass
