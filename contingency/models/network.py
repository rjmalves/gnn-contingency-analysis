from typing import Dict, Tuple, List, Iterator
from os.path import normpath
from os import sep
import numpy as np
import networkx as nx
from networkx.readwrite import read_edgelist
from networkx.readwrite import read_graph6
from itertools import combinations


class Network:

    def __init__(self,
                 name: str,
                 graph: nx.Graph):
        self.__name = name
        self.__graph = graph
        self.__number_of_nodes = graph.number_of_nodes()
        self.__number_of_edges = graph.number_of_edges()
        self.__node_mapping = None
        self.__edge_mapping = None
        self.__reverse_node_mapping = None
        self.__reverse_edge_mapping = None
        self.__valid_contingencies: Dict[int, np.ndarray] = {}
        self.__islanding_contingencies: Dict[int, np.ndarray] = {}

    @staticmethod
    def from_edgelist(filename: str) -> "Network":
        name = normpath(filename).split(sep)[-1].split(".")[0]
        graph = read_edgelist(filename)
        return Network(name, graph)

    @staticmethod
    def from_graph6(filename: str) -> List["Network"]:
        graphs = read_graph6(filename)
        name = normpath(filename).split(sep)[-1].split(".")[0]
        names = [f"{name}_{i}" for i in range(len(graphs))]
        return [Network(n, g) for n, g in zip (names, graphs)]

    @property
    def name(self) -> str:
        return self.__name

    @property
    def graph(self) -> nx.Graph:
        return self.__graph

    @property
    def node_mapping(self) -> Dict[str, int]:

        def __eval_node_mapping() -> Dict[str, int]:
            nodes = list(self.__graph.nodes)
            return {n: i for i, n in enumerate(nodes)}

        if self.__node_mapping is None:
            self.__node_mapping = __eval_node_mapping()
        return self.__node_mapping

    @property
    def edge_mapping(self) -> Dict[Tuple[int, int], int]:

        def __eval_edge_mapping() -> Dict[Tuple[int, int], int]:
            edges = list(self.__graph.edges)
            return {(self.node_mapping[n[0]],
                     self.node_mapping[n[1]]): i
                     for i, n in enumerate(edges)}

        if self.__edge_mapping is None:
            self.__edge_mapping = __eval_edge_mapping()
        return self.__edge_mapping

    def __reverse_mapping(mapping: dict) -> dict:
        return {v: k for k, v in mapping.items()}

    @property
    def reverse_node_mapping(self) -> Dict[int, str]:

        if self.__reverse_node_mapping is None:
            m = Network.__reverse_mapping(self.node_mapping)
            self.__reverse_node_mapping = m
        return self.__reverse_node_mapping

    @property
    def reverse_edge_mapping(self) -> Dict[int, Tuple[str, str]]:

        if self.__reverse_edge_mapping is None:
            m = Network.__reverse_mapping(self.edge_mapping)
            self.__reverse_edge_mapping = {i: (self.node_from_mapping(e[0]),
                                               self.node_from_mapping(e[1]))
                                           for i, e in m.items()}
        return self.__reverse_edge_mapping

    def node_from_mapping(self, index: int) -> str:
        return self.reverse_node_mapping[index]

    def edge_from_mapping(self, index: int) -> Tuple[str, str]:
        return self.reverse_edge_mapping[index]

    def __eval_valid_contingency(self,
                                 edges_indices_to_remove: List[int],
                                 ) -> bool:
        edges_to_remove = [self.edge_from_mapping(i)
                           for i in edges_indices_to_remove]
        self.__graph.remove_edges_from(edges_to_remove)
        conn = nx.is_connected(self.__graph)
        self.__graph.add_edges_from(edges_to_remove)
        return conn

    def __eval_contingencies(self, order: int):
        edge_indices = list(range(self.__number_of_edges))
        edge_combinations = list(combinations(edge_indices, order))
        validities = [self.__eval_valid_contingency(e)
                      for e in edge_combinations]
        valids = [i for i, v in enumerate(validities) if v]
        islandings = [i for i, v in enumerate(validities) if not v]

        valid_matrix = np.zeros((len(valids), order),
                                dtype=np.int32)
        island_matrix = np.zeros((len(islandings), order),
                                 dtype=np.int32)
        for i, vi in enumerate(valids):
            valid_matrix[i, :] = edge_combinations[vi]
        for i, ii in enumerate(islandings):
            island_matrix[i, :] = edge_combinations[ii]

        self.__valid_contingencies[order] = valid_matrix
        self.__islanding_contingencies[order] = island_matrix

    def valid_contingencies(self,
                            order: int
                            ) -> Iterator[List[Tuple[str, str]]]:
        if order not in self.__valid_contingencies:
            self.__eval_contingencies(order)
        mat = self.__valid_contingencies[order]
        for i in range(mat.shape[0]):
            contingency = [self.reverse_edge_mapping[e]
                           for e in mat[i, :]]
            yield contingency

    def islanding_contingencies(self,
                                order: int
                                ) -> Iterator[List[Tuple[str, str]]]:
        if order not in self.__islanding_contingencies:
            self.__eval_contingencies(order)
        mat = self.__islanding_contingencies[order]
        for i in range(mat.shape[0]):
            contingency = [self.reverse_edge_mapping[e]
                           for e in mat[i, :]]
            yield contingency
