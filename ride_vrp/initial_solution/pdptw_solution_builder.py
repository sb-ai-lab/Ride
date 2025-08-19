import dataclasses
import logging
import math
from typing import List, Generic, TypeVar

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np

from ride_vrp.common_model import SolverFactory, Solver, ModelFactory
from ride_vrp.initial_solution_builder import InitialSolutionBuilder
from ride_vrp.routing_manager import RoutingManager, InnerNode

M = TypeVar('M')

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PDPTWSolutionBuilder(InitialSolutionBuilder, Generic[M]):
    solver: Solver[M] | SolverFactory[M]
    model_factory: ModelFactory[M]

    max_problem_size: int = 25
    inverse_weight: bool = False

    def solve_partition(self, routing_manager: RoutingManager) -> list[list[InnerNode]]:
        model = self.model_factory.build_model(routing_manager)
        solver = self.solver
        if isinstance(solver, SolverFactory):
            solver = solver.build_solver(model)
        return solver.solve(model)

    def get_initial_solution(self, routing_manager: RoutingManager) -> List[List[InnerNode]]:
        cg = PDPTWSolutionBuilder.generate_full_graph(routing_manager)

        start2end: dict[int, list[InnerNode]] = {}
        for pd in routing_manager.get_pick_up_and_delivery_nodes():
            a: InnerNode = routing_manager.nodes()[pd[0]]
            b: InnerNode = routing_manager.nodes()[pd[1]]
            cg.add_node(a.id)
            start2end[a.id] = [a, b]

        cg = cg.to_undirected()

        for u, v, d in cg.edges(data=True):
            d['length'] = 1 / (0.0001 + d['length'])

        ucg = cg.to_undirected()
        if nx.is_connected(ucg):
            graphs = [cg]
        else:
            res = []
            for i, c in enumerate(nx.connected_components(cg)):
                res.append(cg.subgraph(c))
            graphs = res

        car2path = {}
        NUM_SOl = 0
        for cg in graphs:
            G: ig.Graph = ig.Graph.from_networkx(cg)
            l, r = 0.1, 128.0
            iterations = 0
            if len(cg.edges) == 0:
                cms = [{u} for u in cg.nodes()]
            else:
                cms = find_cms(G, resolution=(l + r) / 2)
                # cms = nx.community.louvain_communities(cg, weight='length', resolution=(l + r) / 2)
                max_len_cms = max(len(c) for c in cms)
                while max_len_cms > self.max_problem_size or max_len_cms < 20:
                    if max_len_cms > self.max_problem_size:
                        l = (r + l) / 2
                    else:
                        r = (l + r) / 2
                    # cms = nx.community.louvain_communities(cg, weight='length', resolution=(l + r) / 2)
                    cms = find_cms(G, resolution=(l + r) / 2)
                    max_len_cms = max(len(c) for c in cms)
                    log.debug(f"clustering: {max_len_cms, (l + r) / 2}")
                    iterations += 1
                    if iterations == 10:
                        break

            for ii, c in enumerate(cms):
                nodes = [ccc for cc in c for ccc in start2end[cc]]
                cars = [car for car in routing_manager.cars() if car.id not in car2path]

                part = routing_manager.sub_problem(
                    nodes,
                    cars
                )

                solution = self.solve_partition(part)

                for i, s in enumerate(solution):
                    if len(s) > 0:
                        car2path[part.cars()[i].id] = s

        solution = []
        for i, car in enumerate(routing_manager.cars()):
            if car.id in car2path:
                solution.append(car2path[car.id])
            else:
                solution.append([])
        return solution

    @classmethod
    def generate_full_graph(cls, routing_manager: RoutingManager) -> nx.DiGraph:
        cg = nx.DiGraph()

        for pd_indices1 in routing_manager.get_pick_up_and_delivery_nodes():
            pd1: list[InnerNode] = [routing_manager.nodes()[i] for i in pd_indices1]
            for pd_indices2 in routing_manager.get_pick_up_and_delivery_nodes():
                pd2: list[InnerNode] = [routing_manager.nodes()[i] for i in pd_indices2]

                a, b = pd1[0], pd1[1]
                c, d = pd2[0], pd2[1]

                if a.id == c.id:
                    continue

                l0 = routing_manager.get_distance(a, b) + routing_manager.get_distance(c, d) + 0.01
                t0 = routing_manager.get_time(a, b) + routing_manager.get_time(c, d) + 0.01

                l1, t1, remainders = min(
                    get_len([a, b, c, d], routing_manager),

                    get_len([a, c, d, b], routing_manager),
                    get_len([a, c, b, d], routing_manager),

                    key=lambda x: x[0],
                )

                if l1 > 0 and math.isinf(l1):
                    continue
                cost = min((l1 - l0) / l0, 2)
                cost = np.exp(cost)

                if (a.id, c.id) not in cg.edges() or cost < cg.edges()[a.id, c.id]['length']:
                    if (a.id, c.id) in cg.edges():
                        cg.edges()[a.id, c.id]['length'] = min(cost, cg.edges()[a.id, c.id]['length'])
                    else:
                        cg.add_edge(a.id, c.id, length=cost)

                    cg.edges()[a.id, c.id]['l_ab'] = routing_manager.get_distance(a, b)
                    cg.edges()[a.id, c.id]['l_cd'] = routing_manager.get_distance(c, d)
                    cg.edges()[a.id, c.id]['l0'] = l0
                    cg.edges()[a.id, c.id]['l1'] = l1

                    cg.edges()[a.id, c.id]['t_ab'] = routing_manager.get_time(a, b)
                    cg.edges()[a.id, c.id]['t_cd'] = routing_manager.get_time(c, d)
                    cg.edges()[a.id, c.id]['t0'] = t0
                    cg.edges()[a.id, c.id]['t1'] = t1

                    cg.edges()[a.id, c.id]['remainders'] = remainders

        return cg


def find_cms(g: ig.Graph, resolution):
    # Get clustering
    partition = la.find_partition(g,
                                  partition_type=la.CPMVertexPartition,
                                  weights=g.es['length'],  # Исправлено здесь
                                  resolution_parameter=resolution)
    communities = []
    for community in partition:
        node_set = set()
        for v in community:
            node_set.add(g.vs[v]['_nx_name'])
        communities.append(node_set)
    return communities


def get_len(nodes: list[InnerNode], routing_manager: RoutingManager) -> tuple[float, float, list[float]]:
    remainders = [nodes[0].end_time - nodes[0].start_time]
    time = nodes[0].start_time + nodes[0].service_time
    total_length = 0.0
    prev = nodes[0]
    for node in nodes[1:]:
        time = int(time + routing_manager.get_time(prev, node))
        remainders.append(node.end_time - int(time))
        total_length += routing_manager.get_distance(prev, node)
        a, b = node.start_time, node.end_time
        if time > b + 1:
            return float('inf'), float('inf'), []
        time = int(max(time, a) + node.service_time)
        prev = node
    return total_length, float(time), remainders
