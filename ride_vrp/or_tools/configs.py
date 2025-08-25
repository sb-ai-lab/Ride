from dataclasses import dataclass, field

from ortools.constraint_solver import routing_enums_pb2


@dataclass
class ModelConfig:
    max_execution_time_minutes: float = field(default=1)
    max_solution_number: int = field(default=-1)

    ls_type: routing_enums_pb2.LocalSearchMetaheuristic = field(
        default=routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    first_solution_type: routing_enums_pb2.FirstSolutionStrategy = field(
        default=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)

