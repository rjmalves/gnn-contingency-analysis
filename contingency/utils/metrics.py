from typing import Dict
import networkx as nx


def centrality(g: nx.Graph) -> Dict[str, float]:
    return nx.current_flow_betweenness_centrality(g)
