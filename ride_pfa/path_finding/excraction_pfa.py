from networkx import Graph

from ride_pfa.path_finding.pfa import PathFinding, Path, PathFindingCls

__all__ = [
    'ExtractionPfa'
]


class ExtractionPfa(PathFinding):

    def __init__(self, g: Graph, upper: PathFinding,
                 down: PathFindingCls,
                 weight: str = 'weight',
                 cluster: str = 'cluster'):
        super().__init__(g=g, weight=weight)
        self.cluster: str = cluster
        self.upper: PathFinding = upper
        self.down: PathFindingCls = down

    def find_path(self, start: int, end: int) -> Path:
        cluster = self.cluster
        nodes = self.g.nodes()
        c1, c2 = nodes[start][cluster], nodes[end][cluster]
        _, path = self.upper.find_path(start=c1, end=c2)
        return self.down.find_path_cls(start=start, end=end, cms=set(p for p in path))
