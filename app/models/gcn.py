import torch
from torch import nn
from torch_geometric.nn import GCNConv

from app.models.basemodel import BaseModel, ModelFactory


class GCN(BaseModel):
    def __init__(
        self,
        model_name: str = "gcn",
        c_in: int = 1,
        c_out: int = 1,
        d_hidden: int = 48,
    ) -> None:
        super(GCN, self).__init__(model_name)
        self.c_in = c_in
        self.c_out = c_out
        self.d_hidden = d_hidden

        # Constructs the layers
        self.conv1 = GCNConv(c_in, d_hidden)
        self.conv2 = GCNConv(d_hidden, d_hidden)
        self.linear = nn.Linear(d_hidden, c_out)

    def forward(
        self, x: torch.Tensor, edge_index, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.linear(x)
        return x


ModelFactory().register("gcn", GCN)
