import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from contingency.models.network import Network
from contingency.controllers.screener import ExhaustiveScreener


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = [
        "#f76469",
        "#4405f2",
    ]

    for i in range(2):
        ax.scatter(
            z[color == i, 0], z[color == i, 1], s=70, c=cmap[i], label=f"{i}"
        )
    plt.legend()
    plt.tight_layout()
    plt.show()


# Imports edgelist
G = nx.read_edgelist("data/ieee39.txt")
k = 1
net = Network("IEEE", G)
screener = ExhaustiveScreener(net)
deltas = screener.global_deltas(k)
classes = {k: 0 for k in deltas.keys()}
max_delta = max(list(deltas.values()))
for d in deltas.keys():
    if deltas[d] / max_delta >= 0.5:
        classes[d] = 1
    else:
        classes[d] = 0


# Makes the line-graph
Gl = nx.line_graph(G)
# Canonical relabeling
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


# Divide nodes by class
np.random.seed(10)
class_set = set(classes_relabel.values())
nodes_classes = {v: [] for v in class_set}
for n, c in classes_relabel.items():
    nodes_classes[c].append(n)
for c in class_set:
    nodes_classes[c] = np.array(nodes_classes[c])
    np.random.shuffle(nodes_classes[c])


# Divides train-val-test splits
# train_split = 0.6
# val_split = 0.0
# test_split = 1 - (train_split + val_split)
# train_nodes_by_classes = {v: [] for v in class_set}
# val_nodes_by_classes = {v: [] for v in class_set}
# test_nodes_by_classes = {v: [] for v in class_set}
# for c in class_set:
#     nodes = nodes_classes[c]
#     n = len(nodes)
#     train_idx = round(n * train_split)
#     val_idx = train_idx + round(n * val_split)
#     train_nodes_by_classes[c] = nodes[:train_idx]
#     val_nodes_by_classes[c] = nodes[train_idx:val_idx]
#     test_nodes_by_classes[c] = nodes[val_idx:]
# train_nodes = []
# val_nodes = []
# test_nodes = []
# for c in class_set:
#     train_nodes += list(train_nodes_by_classes[c])
#     val_nodes += list(val_nodes_by_classes[c])
#     test_nodes += list(test_nodes_by_classes[c])


# Divides train-val-test splits balancing by class labels
train_split = 0.5
test_split = 1 - train_split
train_nodes_by_classes = {v: [] for v in class_set}
val_nodes_by_classes = {v: [] for v in class_set}
test_nodes_by_classes = {v: [] for v in class_set}
less_elements = min([len(c) for c in nodes_classes.values()])
num_train_elements_by_class = round(train_split * less_elements)
for c in class_set:
    nodes = nodes_classes[c]
    train_nodes_by_classes[c] = nodes[:num_train_elements_by_class]
    test_nodes_by_classes[c] = nodes[num_train_elements_by_class:]
train_nodes = []
val_nodes = []
test_nodes = []
for c in class_set:
    train_nodes += list(train_nodes_by_classes[c])
    test_nodes += list(test_nodes_by_classes[c])


print(f"Train nodes = {train_nodes}")
print(f"Test nodes = {test_nodes}")

# Data object for torch
data = from_networkx(Gn)

# Embedding dimension
n = Gn.number_of_nodes()
d = 32
data.num_features = d
data.num_classes = len(class_set)
data.x = torch.from_numpy(np.random.rand(n, d).astype(np.float32))

# Add labels and masks to data object
train_mask = np.zeros((n,), dtype=np.bool8)
val_mask = np.zeros((n,), dtype=np.bool8)
test_mask = np.zeros((n,), dtype=np.bool8)
for k in train_nodes:
    train_mask[k] = True
for k in val_nodes:
    val_mask[k] = True
for k in test_nodes:
    test_mask[k] = True

y = np.zeros((n,))
for c in class_set:
    for k in nodes_classes[c]:
        y[k] = c

data.train_mask = torch.from_numpy(train_mask)
data.val_mask = torch.from_numpy(val_mask)
data.test_mask = torch.from_numpy(test_mask)
data.y = torch.from_numpy(y.astype(np.int64))


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
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
