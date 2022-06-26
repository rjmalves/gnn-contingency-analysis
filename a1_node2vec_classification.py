from typing import Dict, List
import networkx as nx
import numpy as np
import pandas as pd
import torch
from itertools import product
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import Node2Vec
from torch_geometric.data import DataLoader, Data
import matplotlib.pyplot as plt


GRAPH = "ieee118"
EDGELIST = f"/home/rogerio/git/k-contingency-screening/{GRAPH}.txt"
K = [1]
N_EVALS = 10
TOL = 0.25
EMBEDDING_D = [128]
TRAIN_SPLIT = [0.1]
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


# def generate_labels(
#     deltas: Dict[tuple, float], tol_sup: float
# ) -> Dict[tuple, int]:
#     classes = {k: 0 for k in deltas.keys()}
#     nonzeros = [d for d in list(deltas.values()) if d > 0]
#     max_delta = max(nonzeros)
#     min_delta = min(nonzeros)
#     for d in deltas.keys():
#         if (deltas[d] - min_delta) / (max_delta - min_delta) >= tol_sup:
#             classes[d] = 1
#         elif deltas[d] > 0:
#             classes[d] = 0
#         else:
#             classes[d] = -1
#     return classes


def generate_labels(
    deltas: Dict[tuple, float], quantile: float
) -> Dict[tuple, int]:
    classes = {k: 0 for k in deltas.keys()}
    nonzeros = np.array([d for d in list(deltas.values()) if d > 0])
    threshold = np.quantile(nonzeros, 1 - quantile)
    for d in deltas.keys():
        if deltas[d] >= threshold:
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
    mapas = {antigo: novo for antigo, novo in zip(G.nodes, Gn.nodes)}
    classes_relabel = {}
    for k, v in classes.items():
        edge = (mapas[k[0]], mapas[k[1]])
        if (k[0], k[1]) in classes.keys():
            classes_relabel[edge] = v
        else:
            reverse_edge = (mapas[k[1]], mapas[k[0]])
            classes_relabel[reverse_edge] = v
    return classes_relabel


def divide_edges_in_classes(
    classes_relabel: Dict[tuple, int]
) -> Dict[int, np.ndarray]:
    class_set = set(classes_relabel.values())
    edges_classes = {v: [] for v in class_set}
    for n, c in classes_relabel.items():
        edges_classes[c].append(n)
    for c in class_set:
        edges_classes[c] = np.array(edges_classes[c])
        np.random.shuffle(edges_classes[c])
    return edges_classes


# Divides train-val-test splits balancing by class labels
def split_edges(train_split: float, edges_classes: Dict[int, np.ndarray]):
    class_set = [c for c in list(edges_classes.keys()) if c != -1]
    train_edges_by_classes = {v: [] for v in class_set}
    test_edges_by_classes = {v: [] for v in class_set}
    less_elements = min([len(edges_classes[v]) for v in class_set])
    num_train_elements_by_class = max([1, round(train_split * less_elements)])
    for c in class_set:
        edges = edges_classes[c]
        train_edges_by_classes[c] = edges[:num_train_elements_by_class]
        test_edges_by_classes[c] = edges[num_train_elements_by_class:]
    train_edges = []
    test_edges = []
    for c in class_set:
        train_edges += list(train_edges_by_classes[c])
        test_edges += list(test_edges_by_classes[c])
    print(f"Train: {train_edges}")
    print(f"Test: {test_edges}")
    return train_edges, test_edges


def create_torch_data(
    G: nx.Graph,
    embedding_dimension: int,
) -> Data:
    Gn: nx.Graph = nx.convert_node_labels_to_integers(G)
    data = from_networkx(Gn)

    # Embedding dimension
    n = Gn.number_of_nodes()
    d = embedding_dimension
    data.num_features = d
    data.x = torch.from_numpy(np.random.rand(n, d).astype(np.float32))

    return data


def train(
    model: Node2Vec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def extract_edge_embeddings(
    model: Node2Vec, data: Data
) -> Dict[tuple, np.ndarray]:
    z = model().detach().cpu().numpy()
    edge_embedding = {}
    for u, v in data.edge_index.t():
        edge_embedding[(int(u.numpy()), int(v.numpy()))] = np.multiply(
            z[u], z[v]
        )
    return edge_embedding


def test(
    edge_embeddings: Dict[tuple, np.ndarray],
    embedding_d: int,
    classes: Dict[tuple, int],
    train_split: float,
):
    # Assembles the X and y matrices
    edges_classes = divide_edges_in_classes(classes)
    train_edges, test_edges = split_edges(train_split, edges_classes)
    n_edges_train = len(train_edges)
    n_edges_test = len(test_edges)
    X_train = np.zeros((n_edges_train, embedding_d))
    X_test = np.zeros((n_edges_test, embedding_d))
    y_train = np.zeros((n_edges_train,))
    y_test = np.zeros((n_edges_test,))
    for i in range(n_edges_train):
        edge = (train_edges[i][0], train_edges[i][1])
        X_train[i, :] = edge_embeddings[edge]
        y_train[i] = classes[edge]
    for i in range(n_edges_test):
        edge = (test_edges[i][0], test_edges[i][1])
        X_test[i, :] = edge_embeddings[edge]
        y_test[i] = classes[edge]
    print(f"Y train: {y_train}")
    print(f"Y test: {y_test}")
    # Trains the classifier
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    return classification_report(
        y_test,
        y_hat,
        target_names=["regular", "critical"],  # TODO - padronizar
        output_dict=True,
    )


def process_test_results(
    result_report: dict, k: int, train_split: int, embedding_d: int
):
    result_dict: Dict[str, list] = {
        "k": [k],
        "train_split": [train_split],
        "embedding_d": [embedding_d],
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


combinations = list(product(K, TRAIN_SPLIT, EMBEDDING_D))
G = nx.read_edgelist(EDGELIST)
result = pd.DataFrame()
c = combinations[0]
for c in combinations:
    k, train_split, embedding_d = c
    DELTAS = f"/home/rogerio/git/k-contingency-screening/exaustivo_{GRAPH}_{k}/edge_global_deltas.csv"
    deltas = read_edgelist_deltas(DELTAS)
    print(f"Params = {c}")
    for i in range(1, N_EVALS + 1):
        print(f"Eval {i}")
        name = f"k{k}_split{train_split}_it{i}"
        classes = generate_labels(deltas, TOL)
        classes_relabel = canonical_relabeling(G, classes)

        data = create_torch_data(G, embedding_d)

        model = Node2Vec(
            data.edge_index,
            embedding_dim=embedding_d,
            walk_length=10,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1,
            q=1,
            sparse=True,
        )
        loader = model.loader(
            batch_size=G.number_of_nodes(), shuffle=True, num_workers=4
        )

        optimizer = torch.optim.SparseAdam(
            list(model.parameters()), lr=LEARNING_RATE
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train(model, loader, optimizer)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

        edge_embeddings = extract_edge_embeddings(model, data)

        r = process_test_results(
            test(edge_embeddings, embedding_d, classes_relabel, train_split),
            k,
            train_split,
            embedding_d,
        )
        if result.empty:
            result = r
        else:
            result = pd.concat([result, r], ignore_index=True)

result.to_csv(f"result_a1_{GRAPH}.csv")
