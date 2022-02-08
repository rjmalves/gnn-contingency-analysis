from abc import abstractmethod
from typing import Dict, Tuple
import networkx as nx
import numpy as np
from scipy.special import binom
from copy import deepcopy
from multiprocessing import Pool

from contingency.models.network import Network
from contingency.utils.metrics import centrality


def eval_delta_contingency(graph: nx.Graph,
                           contingency: tuple,
                           reference: Dict[str, float]
                           ) -> float:
        deltas: Dict[str, float] = {}
        g = deepcopy(graph)
        g.remove_edges_from(contingency)
        removal_centrality = centrality(g)
        for k, v in reference.items():
            deltas[k] = abs(removal_centrality[k] - v)
        return sum(list(deltas.values()))


class Screener:

    def __init__(self,
                 network: Network):
        self._network = network
        self._reference_centrality = None

    @property
    def network(self) -> Network:
        return self._network

    @property
    def reference_centrality(self) -> Dict[str, float]:
        if self._reference_centrality is None:
            self._reference_centrality = centrality(self.network.graph)
        return self._reference_centrality

    @abstractmethod
    def deltas(self, order: int) -> Dict[tuple, float]:
        pass

    @abstractmethod
    def global_deltas(self, order: int) -> Dict[Tuple[str, str],
                                                float]:
        pass

    @abstractmethod
    def normalized_global_deltas(self, order) -> Dict[Tuple[str, str],
                                                      float]:
        pass


class ExhaustiveScreener(Screener):

    def __init__(self,
                 network: Network,
                 num_processors: int = 1):
        super().__init__(network)
        self.__num_processors = num_processors
        self.__deltas: Dict[int,
                            Dict[tuple,
                                 float]] = {}
        self.__global_deltas: Dict[int,
                                   Dict[tuple,
                                        float]] = {}

    def __eval_deltas(self, order: int):
        deltas: Dict[tuple, float] = {}
        # Paralelismo
        with Pool(processes=self.__num_processors) as pool:
            f = eval_delta_contingency
            g = self.network.graph
            ref = self.reference_centrality
            async_res = {tuple(c): pool.apply_async(f, 
                                                    (g,
                                                     c,
                                                     ref))
                         for c in self.network.valid_contingencies(order)}
            deltas = {i: r.get(timeout=30)
                      for i, r in async_res.items()}
        self.__deltas[order] = deltas

    # Override
    def deltas(self, order: int) -> Dict[tuple, float]:
        if order not in self.__deltas:
            self.__eval_deltas(order)
        return self.__deltas[order]

    def __eval_global_deltas(self, order: int):
        deltas = self.deltas(order)
        edges = list(self._network.graph.edges)
        global_deltas = {e: 0 for e in edges}
        for contingency, delta in deltas.items():
            for edge in contingency:
                global_deltas[edge] += delta
        self.__global_deltas[order] = global_deltas

    # Override
    def global_deltas(self, order: int) -> Dict[Tuple[str, str],
                                                float]:
        if order not in self.__global_deltas:
            self.__eval_global_deltas(order)
        return self.__global_deltas[order]

    # Override
    def normalized_global_deltas(self, order) -> Dict[Tuple[str, str],
                                                      float]:
        deltas = self.global_deltas(order)
        n = self.network.graph.number_of_nodes()
        m = self.network.graph.number_of_edges()
        factor = binom(m - 1, order - 1) * n
        norm_deltas = {e: d / factor for e, d in deltas.items()}
        return norm_deltas
