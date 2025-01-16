from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NewType, Optional

import networkx as nx

__all__ = [
    "Path",
    "PathMatrix",
    "PathFinding",
    "PathFindingCls"
]

Path = NewType('Path', tuple[float, list[int]])
PathMatrix = NewType('Path', dict[int, tuple[float, int]])


@dataclass
class PathFinding(ABC):
    g: nx.Graph
    weight: str = 'length'

    @abstractmethod
    def find_path(self, start: int, end: int, ) -> Path:
        pass


@dataclass
class PathFindingCls(PathFinding):
    cluster: str = 'cluster'

    def find_path(self, start: int, end: int) -> Path:
        return self.find_path_cls(start=start, end=end, cms=None)

    @abstractmethod
    def find_path_cls(self,
                      start: int,
                      end: int,
                      cms: Optional[set[int]] = None) -> Path:
        pass
