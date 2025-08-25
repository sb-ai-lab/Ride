import numpy as np

from ride_vrp.data_model import Node
from ride_vrp.distance_matrix import DistanceMatrix


class CoordinateDistanceMatrix(DistanceMatrix[object]):

    def __init__(self, nodes: list[Node], scale_dst: float = 1, scale_time: float = 1):
        n = len(nodes)
        self.id2index = {n.id: i for i, n in enumerate(nodes)}
        self.dsts = np.array([[get_dst(nodes[i], nodes[j]) for j in range(n)] for i in range(n)], dtype=np.float32)
        self.scale_dst = scale_dst
        self.scale_time = scale_time

    def get_distance(self, a: object, b: object) -> float:
        return float(self.dsts[self.id2index[a], self.id2index[b]] * self.scale_dst)

    def get_time(self, a: object, b: object) -> float:
        return float(self.dsts[self.id2index[a], self.id2index[b]] * self.scale_time)


def get_dst(node1: Node, node2: Node):
    du = node1.coordinates
    dv = node2.coordinates
    return np.sqrt((du[0] - dv[0]) ** 2 + (du[1] - dv[1]) ** 2)
