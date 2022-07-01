import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from typing import Tuple, Dict, List
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class CGE:
    def __init__(
        self,
        data: Data,
        embedding_dimension: int,
        walk_length: int,
        context_window_size: int,
        walks_per_node: int,
        number_of_negative_samples: int,
        p: int,
        q: int,
    ):
        self.data = data
        self.embedding_dimension = embedding_dimension
        self.embedding_model = Node2Vec(
            data.edge_index,
            embedding_dim=embedding_dimension,
            walk_length=walk_length,
            context_size=context_window_size,
            walks_per_node=walks_per_node,
            num_negative_samples=number_of_negative_samples,
            p=p,
            q=q,
            sparse=True,
        )
        self.classification_model = SVC(random_state=0, probability=True)

    @property
    def edge_embeddings(self) -> Dict[tuple, np.ndarray]:
        z = self.embedding_model().detach().cpu().numpy()
        edge_embedding = {}
        for u, v in self.data.edge_index.t():
            edge_embedding[(int(u.numpy()), int(v.numpy()))] = np.multiply(
                z[u], z[v]
            )
        return edge_embedding

    def edge_embeddings_with_labels(
        self, edges_by_labels: Dict[int, np.ndarray]
    ) -> pd.DataFrame:
        embeddings = self.edge_embeddings
        labels = list(edges_by_labels.keys())
        df = pd.DataFrame()
        embeddings_by_labels: Dict[int, np.ndarray] = {
            label: np.zeros(
                (edges_by_labels[label].shape[0], self.embedding_dimension)
            )
            for label in labels
        }
        for label in labels:
            for i, edge in enumerate(edges_by_labels[label]):
                edge_tuple = (edge[0], edge[1])
                embeddings_by_labels[label][i, :] = embeddings[edge_tuple]

            df_label = pd.DataFrame(
                embeddings_by_labels[label],
                columns=[
                    f"z{d}" for d in range(1, self.embedding_dimension + 1)
                ],
            )
            df_label["label"] = label
            df = pd.concat([df, df_label], ignore_index=True)
        return df

    def train_data(
        self,
        train_edges: List[np.ndarray],
        edges_labels: Dict[Tuple[int, int], int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_edges_train = len(train_edges)
        X_train = np.zeros((n_edges_train, self.embedding_dimension))
        y_train = np.zeros((n_edges_train,))
        for i in range(n_edges_train):
            edge = (train_edges[i][0], train_edges[i][1])
            X_train[i, :] = self.edge_embeddings[edge]
            y_train[i] = edges_labels[edge]
        return X_train, y_train

    def test_data(
        self,
        test_edges: List[np.ndarray],
        edges_labels: Dict[Tuple[int, int], int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_edges_test = len(test_edges)
        X_test = np.zeros((n_edges_test, self.embedding_dimension))
        y_test = np.zeros((n_edges_test,))
        for i in range(n_edges_test):
            edge = (test_edges[i][0], test_edges[i][1])
            X_test[i, :] = self.edge_embeddings[edge]
            y_test[i] = edges_labels[edge]
        return X_test, y_test


def train_embedding(
    model: CGE,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_edges,
    test_edges,
    edges_labels,
):
    model.embedding_model.train()
    loader = model.embedding_model.loader(
        batch_size=data.num_nodes, shuffle=True
    )
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.embedding_model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_classification(model, train_edges, edges_labels)
    X_test, y_test = model.test_data(test_edges, edges_labels)
    yhat = model.classification_model.predict_proba(X_test)

    return total_loss / len(loader), log_loss(y_test, yhat)


def train_classification(
    model: CGE,
    train_edges: List[np.ndarray],
    edges_labels: Dict[Tuple[int, int], int],
):
    X_train, y_train = model.train_data(train_edges, edges_labels)
    model.classification_model.fit(X_train, y_train)


def test(
    model: CGE,
    test_edges: List[np.ndarray],
    edges_labels: Dict[Tuple[int, int], int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_test, y_test = model.test_data(test_edges, edges_labels)
    yhat = model.classification_model.predict_proba(X_test)
    return torch.from_numpy(y_test), torch.from_numpy(yhat)
