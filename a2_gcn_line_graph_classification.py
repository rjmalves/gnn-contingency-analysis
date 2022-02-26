from typing import Dict, List
import networkx as nx
import numpy as np
import pandas as pd
import torch
from itertools import product
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt


GRAPH = "itaipu11"
EDGELIST = f"/home/rogerio/git/k-contingency-screening/{GRAPH}.txt"
K = [1, 2, 3, 4]
N_EVALS = 50
TOL = 0.7
EMBEDDING_D = 128
TRAIN_SPLIT = [0.1, 0.2, 0.3, 0.4, 0.5]
DROPOUT = 0.5
HIDDEN_CHANNELS = [64]
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3


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


def read_edgelist_deltas(arq_deltas: str) -> Dict[tuple, float]:
    deltas = open(arq_deltas, "r")
    lin_deltas = ""
    delta_dict: Dict[tuple, float] = {}
    while True:
        lin_deltas = deltas.readline()
        if len(lin_deltas) == 0:
            break
        src, dst, delta = lin_deltas.split(",")
        delta_dict[(src, dst)] = float(delta)
    return delta_dict


def generate_labels(
    deltas: Dict[tuple, float], tol_sup: float
) -> Dict[tuple, int]:
    classes = {k: 0 for k in deltas.keys()}
    nonzeros = [d for d in list(deltas.values()) if d > 0]
    max_delta = max(nonzeros)
    min_delta = min(nonzeros)
    for d in deltas.keys():
        if (deltas[d] - min_delta) / (max_delta - min_delta) >= tol_sup:
            classes[d] = 1
        elif deltas[d] > 0:
            classes[d] = 0
        else:
            classes[d] = -1
    return classes


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
    return classes_relabel


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
def split_nodes(train_split: float, nodes_classes: Dict[int, np.ndarray]):
    class_set = [c for c in list(nodes_classes.keys()) if c != -1]
    train_nodes_by_classes = {v: [] for v in class_set}
    test_nodes_by_classes = {v: [] for v in class_set}
    less_elements = min([len(c) for c in nodes_classes.values()])
    num_train_elements_by_class = max([1, round(train_split * less_elements)])
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


def create_torch_data(
    Gl: nx.Graph,
    train_nodes: List[int],
    test_nodes: List[int],
    nodes_classes: Dict[int, np.ndarray],
    embedding_dimension: int,
) -> Data:
    Gn: nx.Graph = nx.convert_node_labels_to_integers(Gl)
    data = from_networkx(Gn)

    class_set = list(nodes_classes.keys())

    # Embedding dimension
    n = Gn.number_of_nodes()
    d = embedding_dimension
    data.num_features = d
    data.num_classes = len(class_set)
    data.x = torch.from_numpy(np.random.rand(n, d).astype(np.float32))

    # Add labels and masks to data object
    train_mask = np.zeros((n,), dtype=np.bool8)
    test_mask = np.zeros((n,), dtype=np.bool8)
    for k in train_nodes:
        train_mask[k] = True
    for k in test_nodes:
        test_mask[k] = True

    y = np.zeros((n,))
    for c in class_set:
        for k in nodes_classes[c]:
            y[k] = c

    data.train_mask = torch.from_numpy(train_mask)
    data.test_mask = torch.from_numpy(test_mask)
    data.y = torch.from_numpy(y.astype(np.int64))
    return data


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_channels: int,
        num_outputs: int,
        dropout: float,
    ):
        super().__init__()
        self._dropout = dropout
        self.conv1 = GCNConv(num_inputs, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv3(x, edge_index)
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
        out[data.train_mask], data.y[data.train_mask]
    )  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test(model: GCN, data: Data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    # Use the class with highest probability.
    y = np.array(data.y[data.test_mask])
    yhat = np.array(pred[data.test_mask])
    return classification_report(
        y,
        yhat,
        target_names=["regular", "critical"],  # TODO - padronizar
        output_dict=True,
    )


def process_test_results(
    result_report: dict,
    k: int,
    train_split: int,
    channels: int,
):
    result_dict: Dict[str, list] = {
        "k": k,
        "train_split": [train_split],
        "channels": [channels],
    }
    classes = [
        "regular",
        "critical",
        # "non-critical",
        "macro avg",
        "weighted avg",
    ]
    measures = ["precision", "recall", "f1-score", "support"]
    for c in classes:
        for m in measures:
            col = f"{c}_{m}"
            result_dict[col] = [result_report[c][m]]
    result_dict["accuracy"] = [result_report["accuracy"]]
    return pd.DataFrame(data=result_dict)


combinations = list(product(K, TRAIN_SPLIT, HIDDEN_CHANNELS))
G = nx.read_edgelist(EDGELIST)
result = pd.DataFrame()
for c in combinations:
    k, train_split, channels = c
    DELTAS = f"/home/rogerio/git/k-contingency-screening/exaustivo_{GRAPH}_{k}/edge_global_deltas.csv"
    deltas = read_edgelist_deltas(DELTAS)
    print(f"Params = {c}")
    for i in range(1, N_EVALS + 1):
        print(f"Eval {i}")
        name = f"k{k}_split{train_split}_channels{channels}_it{i}"
        classes = generate_labels(deltas, TOL)
        Gl = nx.line_graph(G)
        classes_relabel = canonical_relabeling(Gl, classes)

        nodes_classes = divide_nodes_in_classes(classes_relabel)
        train_nodes, test_nodes = split_nodes(train_split, nodes_classes)
        print(f"Train: {train_nodes}")
        print(f"Test: {test_nodes}")
        data = create_torch_data(
            Gl, train_nodes, test_nodes, nodes_classes, EMBEDDING_D
        )

        model = GCN(
            num_inputs=data.num_features,
            hidden_channels=channels,
            num_outputs=data.num_classes,
            dropout=DROPOUT,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train(model, data, optimizer, criterion)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

        r = process_test_results(
            test(model, data),
            k,
            train_split,
            channels,
        )
        if result.empty:
            result = r
        else:
            result = pd.concat([result, r], ignore_index=True)

result.to_csv(f"result_a2_{GRAPH}.csv")
