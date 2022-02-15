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
from contingency.models.network import Network
from contingency.controllers.screener import ExhaustiveScreener


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


def generate_labels(deltas: Dict[tuple, float]) -> Dict[tuple, float]:
    classes = {k: v for k, v in deltas.items()}

    # max_delta = max(list(deltas.values()))
    # for c in classes.keys():
    #     classes[c] /= max_delta
    return classes


def canonical_relabeling(
    g: nx.Graph, classes: Dict[tuple, float]
) -> Dict[tuple, float]:
    Gn: nx.Graph = nx.convert_node_labels_to_integers(Gl)
    mapas = {novo: antigo for antigo, novo in zip(Gl.nodes, Gn.nodes)}
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


# Divides train-val-test splits
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
    classes: Dict[int, float],
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
        self.linear1 = Linear(hidden_channels, 4)
        self.linear2 = Linear(4, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
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


GRAPH = "data/ieee57.txt"
k = 1
N_EVALS = 50
EMBEDDING_D = [2, 4, 8, 16, 32, 64, 128]
CHANNELS = [4, 8, 16, 32, 64, 128]
TRAIN_SPLIT = [0.2, 0.3, 0.4, 0.5]
NUM_EPOCHS = 200
LEARNING_RATE = 1e-2


combinations = list(product(EMBEDDING_D, CHANNELS, TRAIN_SPLIT))
G = nx.read_edgelist(GRAPH)
net = Network("IEEE", G)
screener = ExhaustiveScreener(net)
deltas = screener.global_deltas(k)

result = pd.DataFrame()
for c in combinations:
    embedding_d, channels, train_split = c
    print(f"Params = {c}")
    for i in range(1, N_EVALS + 1):
        print(f"Eval {i}")
        name = f"k{k}_emb{embedding_d}" + f"split{train_split}_it{i}"
        classes = generate_labels(deltas)
        Gl = nx.line_graph(G)
        classes_relabel, nodes = canonical_relabeling(Gl, classes)

        train_nodes, test_nodes = split_nodes(train_split, nodes)
        data = create_torch_data(
            Gl, train_nodes, test_nodes, classes_relabel, embedding_d
        )

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
        # model.eval()
        # out = model(data.x, data.edge_index)
        # visualize(out, color=data.y, name=name)

result.to_csv(f"result_k{k}_regression.csv")
