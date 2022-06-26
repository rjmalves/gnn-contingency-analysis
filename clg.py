from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
from itertools import product
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from refactor.approaches.labeling import QuantileLabeling
from refactor.approaches.preprocessing import Preprocessing
from refactor.approaches.clg import CLG, train

plt.rcParams["font.size"] = "12"


GRAPH = "ieee300"
EDGELIST = f"/home/rogerio/git/k-contingency-screening/{GRAPH}.txt"
K = [1, 2, 3]
N_EVALS = 50
TOL = [0.05, 0.10, 0.25]
EMBEDDING_D = 128
TRAIN_SPLIT = [0.1]
DROPOUT = 0.1
HIDDEN_CHANNELS = [64]
NUM_EPOCHS = 500
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


def test(model: CLG, data: Data) -> Tuple[np.ndarray, np.ndarray]:
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


def test_predictions(model: CLG, data: Data) -> dict:
    model.eval()
    out = model(data.x, data.edge_index)
    pred = F.softmax(out[:, :2], dim=1).detach().numpy()
    # Use the class with highest probability.
    y = np.array(data.y[data.test_mask])
    yhat = np.array(pred[data.test_mask][:, 1])
    fpr, tpr, thresholds = roc_curve(y, yhat)
    auc = roc_auc_score(y, yhat)
    return {"fpr": fpr, "tpr": tpr, "auc": auc}


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


def process_test_results_predictions(
    result_report: dict,
    k: int,
    tol: float,
    train_split: int,
    channels: int,
):
    result_dict: Dict[str, list] = {
        "k": k,
        "tol": tol,
        "train_split": [train_split],
        "channels": [channels],
    }
    result_dict["auc"] = [result_report["auc"]]
    return pd.DataFrame(data=result_dict)


combinations = list(product(K, TOL, TRAIN_SPLIT, HIDDEN_CHANNELS))
G = nx.read_edgelist(EDGELIST)
result = pd.DataFrame()
for c in combinations:
    k, tol, train_split, channels = c
    DELTAS = f"/home/rogerio/git/k-contingency-screening/exaustivo_{GRAPH}_{k}/edge_global_deltas.csv"
    preprocessor = Preprocessing(
        G,
        DELTAS,
        train_split=train_split,
        embedding_dimension=EMBEDDING_D,
        labeling_strategy=QuantileLabeling(tol),
    )
    print(f"Params = {c}")
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(1, N_EVALS + 1):
        print(f"Eval {i}")
        name = f"k{k}_split{train_split}_channels{channels}_it{i}"
        data = preprocessor.torch_data

        model = CLG(
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

        preds = test_predictions(model, data)
        ax.stairs(
            preds["tpr"],
            np.concatenate((np.array([0.0]), preds["fpr"])),
            alpha=0.2,
            color="red",
            baseline=None,
        )
        r = process_test_results_predictions(
            preds, k, tol, train_split, channels
        )
        if result.empty:
            result = r
        else:
            result = pd.concat([result, r], ignore_index=True)
    ax.plot([0, 1], [0, 1], linestyle="dashed", color="grey", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    plt.tight_layout()
    plt.savefig(f"./figures/{GRAPH}_{k}_{tol}_{train_split}_{channels}.png")
    plt.clf()

result.to_csv(f"result_a2_{GRAPH}_auc.csv")
