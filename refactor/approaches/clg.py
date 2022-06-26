import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


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
        self.conv1 = GCNConv(num_inputs, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


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
    return loss
