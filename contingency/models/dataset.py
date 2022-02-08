import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import pandas as pd
from typing import Union, List, Tuple, Dict, Callable

from contingency.controllers.screener import Screener


class Dataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 num_nodes: int,
                 transform=None,
                 pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_nodes = num_nodes

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.num_nodes}nodes2c.g6"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    
    @staticmethod
    def __node_pair_metrics() -> Dict[str, Callable]:
        return {
            "Degree": nx.degree_centrality,
            "EigenvectorCentrality": nx.eigenvector_centrality_numpy,
            "KatzCentrality": nx.katz_centrality_numpy,
            "ClosenessCentrality": nx.closeness_centrality,
            "CurrentFlowCloseness": nx.current_flow_closeness_centrality,
            "BetweennessCentrality": nx.betweenness_centrality,
            "CommBetweenness": nx.communicability_betweenness_centrality
        }

    @staticmethod
    def __edge_metrics() -> Dict[str, Callable]:
        return {
            "EdgeBetweenness": nx.edge_betweenness_centrality,
            "EdgeCFB": nx.edge_current_flow_betweenness_centrality,
            "EdgeLoadCentrality": nx.edge_load_centrality
            }

    def __eval_node_pair_metrics(self, G: nx.Graph) -> pd.DataFrame:
        metrics = Dataset.__node_pair_metrics()
        indices = []
        results = {e: [] for e in G.edges}
        for n, m in metrics.items():
            indices.append(n)
            metric_value = m(G)
            for e in G.edges:
                # Computes the average of the metric value for
                # both nodes on the edge
                e_m = 0.5 * (metric_value[e[0]] + metric_value[e[1]])
                results[e].append(e_m)
        # Makes the DF for viewing the results
        df_result = pd.DataFrame(data=results,
                                 index=indices)
        return df_result

    def __eval_edge_metrics(self, G: nx.Graph) -> pd.DataFrame:
        metrics = Dataset.__edge_metrics()
        indices = []
        results = {e: [] for e in G.edges}
        for n, m in metrics.items():
            indices.append(n)
            metric_value = m(G)
            for e in G.edges:
                if e not in metric_value:
                    e_m = (e[1], e[0])
                else:
                    e_m = e
                results[e].append(metric_value[e_m])
        # Makes the DF for viewing the results
        df_result = pd.DataFrame(data=results,
                                 index=indices)
        return df_result

    def __eval_metrics(self, G: nx.Graph) -> pd.DataFrame:
        return pd.concat([self.__eval_node_pair_metrics(G),
                          self.__eval_edge_metrics(G)]).T

    def process(self):
        # Lê os grafos do arquivo
        graphs = nx.read_graph6(self.raw_file_names[0])
        # Calcula os dados de interesse
        graph_data = [self.__eval_metrics(G) for G in graphs]
        
        print(graph_data[0])
        # Convete para objetos "Data"
        data_list = []
        for g, d in zip(graphs, graph_data):
            # Node features

            data = Data()


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
