from typing import Protocol, TypeVar

M_cov = TypeVar('M_cov', covariant=True)


class DistanceMatrix(Protocol[M_cov]):

    def get_distance(self, a: M_cov, b: M_cov) -> float:
        pass

    def get_time(self, a: M_cov, b: M_cov) -> float:
        pass
