import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from typing import Tuple, Dict


class CLG(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_channels: int,
        num_outputs: int,
        dropout: float,
    ):
        super().__init__()
        self._dropout = dropout
        self._hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_inputs, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv2(x, edge_index)
        self.node_embeddings = x.detach().numpy()
        x = x.relu()
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.linear(x)
        return x

    def node_embeddings_with_labels(
        self, nodes_by_labels: Dict[int, np.ndarray]
    ) -> pd.DataFrame:
        embeddings = self.node_embeddings
        labels = list(nodes_by_labels.keys())
        df = pd.DataFrame()
        embeddings_by_labels = {
            label: np.zeros(
                (nodes_by_labels[label].shape[0], self._hidden_channels)
            )
            for label in labels
        }
        for label in labels:
            for i, node in enumerate(nodes_by_labels[label]):
                embeddings_by_labels[label][i, :] = embeddings[node, :]

            df_label = pd.DataFrame(
                embeddings_by_labels[label],
                columns=[f"z{d}" for d in range(1, self._hidden_channels + 1)],
            )
            df_label["label"] = label
            df = pd.concat([df, df_label], ignore_index=True)
        return df


def train(
    model: CLG,
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

    val_loss = criterion(out[data.test_mask], data.y[data.test_mask])

    return loss.item(), val_loss.item()


def test(model: CLG, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    out = model(data.x, data.edge_index)
    return data.y[data.test_mask], out[data.test_mask]
