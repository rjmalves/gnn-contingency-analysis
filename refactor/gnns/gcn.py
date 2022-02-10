import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv


def visualize(h, color, nome):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = [
        "#f76469",
        "#fcba03",
        "#5bdb16",
        "#05f2a3",
        "#05e2f2",
        "#4405f2",
        "#f2a2e3",
    ]

    for i in range(7):
        ax.scatter(
            z[color == i, 0], z[color == i, 1], s=70, c=cmap[i], label=f"{i}"
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{nome}.svg")


dataset = Planetoid(
    root="./data/Planetoid", name="Cora", transform=NormalizeFeatures()
)
data = dataset[0]  # Get the first graph object.
data
dataset.num_classes

x: torch.Tensor = data.x
x.dtype


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
n_epochs = 100


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(
        out[data.train_mask], data.y[data.train_mask]
    )  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = (
        pred[data.test_mask] == data.y[data.test_mask]
    )  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(
        data.test_mask.sum()
    )  # Derive ratio of correct predictions.
    return test_acc


for epoch in range(1, n_epochs + 1):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


test_acc = test()
print(f"Test Accuracy: {test_acc:.4f}")
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
