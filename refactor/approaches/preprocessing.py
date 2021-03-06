import networkx as nx
import numpy as np
from typing import Dict, Tuple, List
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from refactor.approaches.labeling import AbstractLabeling, ThresholdLabeling


class Preprocessing:
    def __init__(
        self,
        graph: nx.Graph,
        criticality_edge_file: str,
        train_split: float = 0.10,
        embedding_dimension: int = 32,
        labeling_strategy: AbstractLabeling = ThresholdLabeling(0.7),
    ) -> None:
        self.__graph = graph
        self.__line_graph = nx.line_graph(graph)
        self.__criticality_edge_file = criticality_edge_file
        self.__train_split = train_split
        self.__embedding_dimension = embedding_dimension
        self.__labeling_strategy = labeling_strategy
        self.__criticality = None
        self.__nodes_labels = None
        self.__edges_labels = None
        self.__nodes_by_label = None
        self.__edges_by_label = None
        self.__train_test_nodes = None
        self.__train_test_edges = None
        self.__torch_node_data = None
        self.__torch_edge_data = None

    def __read_criticality(self) -> Dict[tuple, float]:
        criticality = open(self.__criticality_edge_file, "r")
        lin_criticality = ""
        criticality_dict: Dict[tuple, float] = {}
        while True:
            lin_criticality = criticality.readline()
            if len(lin_criticality) == 0:
                break
            src, dst, delta = lin_criticality.split(",")
            criticality_dict[(src, dst)] = float(delta)
        return criticality_dict

    def __generate_labels(self) -> Dict[tuple, int]:
        return self.__labeling_strategy.label(self.criticality)

    def __canonical_relabeling_nodes(
        self, classes: Dict[tuple, int]
    ) -> Dict[tuple, int]:
        Gn: nx.Graph = nx.convert_node_labels_to_integers(self.line_graph)
        maps = {
            novo: antigo
            for antigo, novo in zip(self.line_graph.nodes, Gn.nodes)
        }
        maps = {
            k: tuple([str(n) for n in sorted([int(n) for n in v])])
            for k, v in maps.items()
        }
        canonical_classes = {}
        for k, v in maps.items():
            if v in classes.keys():
                canonical_classes[k] = classes[v]
            else:
                canonical_classes[k] = classes[(v[1], v[0])]
        return canonical_classes

    def __canonical_relabeling_edges(
        self, classes: Dict[tuple, int]
    ) -> Dict[tuple, int]:
        Gn: nx.Graph = nx.convert_node_labels_to_integers(self.__graph)
        mapas = {
            antigo: novo for antigo, novo in zip(self.graph.nodes, Gn.nodes)
        }
        canonical_classes = {}
        for k, v in classes.items():
            edge = (mapas[k[0]], mapas[k[1]])
            if (k[0], k[1]) in classes.keys():
                canonical_classes[edge] = v
            else:
                reverse_edge = (mapas[k[1]], mapas[k[0]])
                canonical_classes[reverse_edge] = v
        return canonical_classes

    def __divide_nodes_in_classes(
        self, classes: Dict[tuple, int]
    ) -> Dict[int, np.ndarray]:
        class_set = set(classes.values())
        nodes_classes = {v: [] for v in class_set}
        for n, c in classes.items():
            nodes_classes[c].append(n)
        for c in class_set:
            nodes_classes[c] = np.array(nodes_classes[c])
            np.random.shuffle(nodes_classes[c])
        return nodes_classes

    def __divide_edges_in_classes(
        self, classes: Dict[tuple, int]
    ) -> Dict[int, np.ndarray]:
        class_set = set(classes.values())
        edges_classes = {v: [] for v in class_set}
        for n, c in classes.items():
            edges_classes[c].append(n)
        for c in class_set:
            edges_classes[c] = np.array(edges_classes[c])
            np.random.shuffle(edges_classes[c])
        return edges_classes

    def __split_nodes(
        self, nodes_classes: Dict[int, np.ndarray]
    ) -> Tuple[List[int], List[int]]:
        class_set = [c for c in list(nodes_classes.keys()) if c != -1]
        train_nodes_by_classes = {v: [] for v in class_set}
        test_nodes_by_classes = {v: [] for v in class_set}
        less_elements = min([len(nodes_classes[v]) for v in class_set])
        num_train_elements_by_class = max(
            [1, round(self.__train_split * less_elements)]
        )
        for c in class_set:
            nodes = nodes_classes[c]
            train_nodes_by_classes[c] = nodes[:num_train_elements_by_class]
            test_nodes_by_classes[c] = nodes[num_train_elements_by_class:]
        train_nodes = []
        test_nodes = []
        for c in class_set:
            train_nodes += list(train_nodes_by_classes[c])
            test_nodes += list(test_nodes_by_classes[c])
        return train_nodes, test_nodes

    def __split_edges(self, edges_classes: Dict[int, np.ndarray]):
        class_set = [c for c in list(edges_classes.keys()) if c != -1]
        train_edges_by_classes = {v: [] for v in class_set}
        test_edges_by_classes = {v: [] for v in class_set}
        less_elements = min([len(edges_classes[v]) for v in class_set])
        num_train_elements_by_class = max(
            [1, round(self.__train_split * less_elements)]
        )
        for c in class_set:
            edges = edges_classes[c]
            train_edges_by_classes[c] = edges[:num_train_elements_by_class]
            test_edges_by_classes[c] = edges[num_train_elements_by_class:]
        train_edges = []
        test_edges = []
        for c in class_set:
            train_edges += list(train_edges_by_classes[c])
            test_edges += list(test_edges_by_classes[c])
        return train_edges, test_edges

    def __generate_torch_node_data(self) -> Data:
        Gn: nx.Graph = nx.convert_node_labels_to_integers(self.line_graph)
        data = from_networkx(Gn)

        class_set = list(self.nodes_by_label.keys())

        # Embedding dimension
        n = Gn.number_of_nodes()
        d = self.__embedding_dimension
        data.num_features = d
        data.num_classes = len(class_set)
        data.x = torch.from_numpy(np.random.rand(n, d).astype(np.float32))

        # Add labels and masks to data object
        train_mask = np.zeros((n,), dtype=np.bool8)
        test_mask = np.zeros((n,), dtype=np.bool8)
        train_nodes, test_nodes = self.train_test_nodes
        for k in train_nodes:
            train_mask[k] = True
        for k in test_nodes:
            test_mask[k] = True

        y = np.zeros((n,))
        for c in class_set:
            for k in self.nodes_by_label[c]:
                y[k] = c

        data.train_mask = torch.from_numpy(train_mask)
        data.test_mask = torch.from_numpy(test_mask)
        data.y = torch.from_numpy(y.astype(np.int64))
        return data

    def __generate_torch_edge_data(self) -> Data:
        Gn: nx.Graph = nx.convert_node_labels_to_integers(self.__graph)
        data = from_networkx(Gn)

        # Embedding dimension
        n = Gn.number_of_nodes()
        d = self.__embedding_dimension
        data.num_features = d
        data.x = torch.from_numpy(np.random.rand(n, d).astype(np.float32))

        return data

    def shuffle(self):
        self.__nodes_by_label = None
        self.__train_test_nodes = None
        self.__torch_node_data = None

    @property
    def graph(self) -> nx.Graph:
        return self.__graph

    @property
    def line_graph(self) -> nx.Graph:
        return self.__line_graph

    @property
    def criticality(self) -> Dict[tuple, float]:
        if self.__criticality is None:
            self.__criticality = self.__read_criticality()
        return self.__criticality

    @property
    def nodes_labels(self) -> Dict[tuple, int]:
        if self.__nodes_labels is None:
            classes = self.__generate_labels()
            self.__nodes_labels = self.__canonical_relabeling_nodes(classes)
        return self.__nodes_labels

    @property
    def edges_labels(self) -> Dict[tuple, int]:
        if self.__edges_labels is None:
            classes = self.__generate_labels()
            self.__edges_labels = self.__canonical_relabeling_edges(classes)
        return self.__edges_labels

    @property
    def nodes_by_label(self) -> Dict[int, np.ndarray]:
        if self.__nodes_by_label is None:
            self.__nodes_by_label = self.__divide_nodes_in_classes(
                self.nodes_labels
            )
        return self.__nodes_by_label

    @property
    def edges_by_label(self) -> Dict[int, np.ndarray]:
        if self.__edges_by_label is None:
            self.__edges_by_label = self.__divide_edges_in_classes(
                self.edges_labels
            )
        return self.__edges_by_label

    @property
    def train_test_nodes(self) -> Tuple[List[int], List[int]]:
        if self.__train_test_nodes is None:
            self.__train_test_nodes = self.__split_nodes(self.nodes_by_label)
        return self.__train_test_nodes

    @property
    def train_test_edges(self) -> Tuple[List[int], List[int]]:
        if self.__train_test_edges is None:
            self.__train_test_edges = self.__split_edges(self.edges_by_label)
        return self.__train_test_edges

    @property
    def torch_node_data(self) -> Data:
        if self.__torch_node_data is None:
            self.__torch_node_data = self.__generate_torch_node_data()
        return self.__torch_node_data

    @property
    def torch_edge_data(self) -> Data:
        if self.__torch_edge_data is None:
            self.__torch_edge_data = self.__generate_torch_edge_data()
        return self.__torch_edge_data
