import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GINConv, Linear


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


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        mlp = MLP(hidden_channels, dataset.num_classes)
        self.conv1 = GINConv(mlp)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.7, training=self.training)
        return x


model = GIN(hidden_channels=128)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
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


def test(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(
        mask.sum()
    )  # Derive ratio of correct predictions.
    return acc


for epoch in range(1, 101):
    loss = train()
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
    )


test_acc = test()
print(f"Test Accuracy: {test_acc:.4f}")
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
