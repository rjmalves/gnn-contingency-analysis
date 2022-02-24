from typing import Dict, List
import networkx as nx
import numpy as np
import pandas as pd
import torch
from itertools import product
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt


GRAPH = "ieee57"
EDGELIST = f"data/{GRAPH}.txt"
K = [1, 2, 3, 4]
N_EVALS = 50
EMBEDDING_D = [2, 4, 8, 16, 32, 64, 128]
TRAIN_SPLIT = [0.1, 0.2, 0.3, 0.4, 0.5]
DROPOUT = 0.5
CHANNELS = [4, 8, 16, 32, 64, 128]
NUM_EPOCHS = 200
LEARNING_RATE = 1e-2


def visualize(h, color, name: str):
    z = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(
        h.detach().cpu().numpy()
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = [
        "#828583",
        "#f76469",
        "#32a852",
    ]

    for i in range(3):
        ax.scatter(
            z[color == i, 0], z[color == i, 1], s=70, c=cmap[i], label=f"{i}"
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png")
    plt.clf()
    plt.close(fig)


def read_edgelist_deltas(
    arq_edgelist: str, arq_deltas: str
) -> Dict[tuple, float]:
    edgelist = open(arq_edgelist, "r")
    deltas = open(arq_deltas, "r")
    lin_edges = ""
    lin_deltas = ""
    delta_dict: Dict[tuple, float] = {}
    while True:
        lin_edges = edgelist.readline()
        lin_deltas = deltas.readline()
        if len(lin_edges) == 0:
            break
        elif " " in lin_edges:
            vertices = lin_edges.split(" ")
        elif "\t" in lin_edges:
            vertices = lin_edges.split("\t")
        else:
            raise RuntimeError("Erro na leitura da lista de arestas")
        v = [v.strip() for v in vertices if len(v) > 0]
        delta_dict[(v[0], v[1])] = float(lin_deltas)
    return delta_dict


def generate_labels(deltas: Dict[tuple, float]) -> Dict[tuple, int]:
    return deltas


def canonical_relabeling(
    g: nx.Graph, classes: Dict[tuple, int]
) -> Dict[tuple, int]:
    Gn: nx.Graph = nx.convert_node_labels_to_integers(g)
    mapas = {novo: antigo for antigo, novo in zip(g.nodes, Gn.nodes)}
    mapas = {
        k: tuple([str(n) for n in sorted([int(n) for n in v])])
        for k, v in mapas.items()
    }
    classes_relabel = {}
    for k, v in mapas.items():
        if v in classes.keys():
            classes_relabel[k] = classes[v]
        else:
            classes_relabel[k] = classes[(v[1], v[0])]
    return classes_relabel, list(Gn.nodes())


def divide_nodes_in_classes(
    classes_relabel: Dict[tuple, int]
) -> Dict[int, np.ndarray]:
    class_set = set(classes_relabel.values())
    nodes_classes = {v: [] for v in class_set}
    for n, c in classes_relabel.items():
        nodes_classes[c].append(n)
    for c in class_set:
        nodes_classes[c] = np.array(nodes_classes[c])
        np.random.shuffle(nodes_classes[c])
    return nodes_classes


# Divides train-val-test splits balancing by class labels
def split_nodes(train_split: float, nodes: List[int]):
    num_train_elements = round(train_split * len(nodes))
    nodes_for_split = np.array(nodes)
    np.random.shuffle(nodes_for_split)
    train_nodes = nodes_for_split[:num_train_elements]
    test_nodes = nodes_for_split[num_train_elements:]
    return train_nodes, test_nodes


def create_torch_data(
    Gl: nx.Graph,
    train_nodes: List[int],
    test_nodes: List[int],
    embedding_dimension: int,
) -> Data:
    Gn: nx.Graph = nx.convert_node_labels_to_integers(Gl)
    data = from_networkx(Gn)

    # Embedding dimension
    n = Gn.number_of_nodes()
    d = embedding_dimension
    data.num_features = d
    data.x = torch.from_numpy(np.random.rand(n, d).astype(np.float32))

    # Add labels and masks to data object
    train_mask = np.zeros((n,), dtype=np.bool8)
    test_mask = np.zeros((n,), dtype=np.bool8)
    for k in train_nodes:
        train_mask[k] = True
    for k in test_nodes:
        test_mask[k] = True

    y = np.zeros((n,))
    for i, (k, v) in enumerate(classes.items()):
        y[i] = v

    data.train_mask = torch.from_numpy(train_mask)
    data.test_mask = torch.from_numpy(test_mask)
    data.y = torch.from_numpy(y.astype(np.float32))
    return data


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_channels: int,
        num_outputs: int,
    ):
        super().__init__()
        self.conv1 = GCNConv(num_inputs, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels, 4)
        self.linear2 = Linear(4, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def train(
    model: GCN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(
        out[data.train_mask], data.y[data.train_mask].unsqueeze(1)
    )  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(model: GCN, data: Data) -> np.ndarray:
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    y = np.array(data.y[data.test_mask])
    yhat = out[data.test_mask].detach().numpy()
    return y, yhat


def process_test_results(
    y: np.ndarray,
    y_hat: np.ndarray,
    embedding_d: int,
    channels: int,
    train_split: int,
):
    result_dict: Dict[str, list] = {
        "embedding_d": [embedding_d],
        "channels": [channels],
        "train_split": [train_split],
        "mse": [mean_squared_error(y, y_hat)],
        "mae": [mean_absolute_error(y, y_hat)],
        "r2": [r2_score(y, y_hat)],
    }
    return pd.DataFrame(data=result_dict)


combinations = list(product(K, EMBEDDING_D, TRAIN_SPLIT, CHANNELS))
G = nx.read_edgelist(EDGELIST)
result = pd.DataFrame()
for c in combinations:
    k, embedding_d, train_split, channels = c
    DELTAS = f"data/exaustivo/exaustivo_{GRAPH}_{k}/edge_global_deltas.csv"
    deltas = read_edgelist_deltas(EDGELIST, DELTAS)
    print(f"Params = {c}")
    for i in range(1, N_EVALS + 1):
        print(f"Eval {i}")
        name = (
            f"k{k}_emb{embedding_d}_"
            + f"split{train_split}_channels{channels}_it{i}"
        )
        classes = generate_labels(deltas)
        Gl = nx.line_graph(G)
        classes_relabel, nodes = canonical_relabeling(Gl, classes)

        train_nodes, test_nodes = split_nodes(train_split, nodes)
        data = create_torch_data(Gl, train_nodes, test_nodes, embedding_d)

        model = GCN(
            num_inputs=data.num_features,
            hidden_channels=channels,
            num_outputs=1,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
        )
        criterion = torch.nn.MSELoss()
        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train(model, data, optimizer, criterion)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

        y, y_hat = test(model, data)
        r = process_test_results(
            y,
            y_hat,
            embedding_d,
            channels,
            train_split,
        )
        if result.empty:
            result = r
        else:
            result = pd.concat([result, r], ignore_index=True)

result.to_csv(f"result_a3_{GRAPH}.csv")
