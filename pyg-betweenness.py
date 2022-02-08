import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch.nn import Sequential, ReLU, Linear
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SimpleDataset(InMemoryDataset):

    def __init__(self,
                 G: nx.Graph,
                 transform=None):
        
        super(SimpleDataset, self).__init__('.',
                                            transform,
                                            None,
                                            None)

        adj_csr: csr_matrix = nx.to_scipy_sparse_matrix(G)
        adj: coo_matrix = adj_csr.tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # Armazena informações do grafo
        data = Data(edge_index=edge_index)
        n_nodes = G.number_of_nodes()
        data.num_nodes = n_nodes

        # Embeddings
        embeddings = np.array(list(dict(G.degree()).values()))
        scale = StandardScaler()
        embeddings = scale.fit_transform(embeddings.reshape(-1, 1))
        data.x = torch.from_numpy(embeddings).type(torch.float32)
        
        # Labels
        targets = list(dict(nx.betweenness_centrality(G)).values())
        labels = np.squeeze(scale.fit_transform(np.asarray(targets).reshape(-1, 1)))
        y = torch.from_numpy(labels).type(torch.float32)
        data.y = y.clone().detach()

        # Dividindo os dados em treino e testes
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(G.nodes()), 
                                                            pd.Series(labels),
                                                            test_size=0.30, 
                                                            random_state=42)

        # Máscaras para filtrar os dados de treino e teste
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data["train_mask"] = train_mask
        data["test_mask"] = test_mask

        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


dataset = 'Cora'
# dataset = SimpleDataset(G)
dataset = Planetoid(".", dataset, transform=T.NormalizeFeatures())
data = dataset[0]
G = to_networkx(data, to_undirected=True)
targets = list(dict(nx.betweenness_centrality(G)).values())
scale = StandardScaler()
labels = np.squeeze(scale.fit_transform(np.asarray(targets).reshape(-1, 1)))
y = torch.from_numpy(labels).type(torch.float32)
data.y = y.clone().detach()
data["train_mask"]

len(data.x)

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt


# GCN model with 2 layers 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.mlp = Sequential(
            Linear(64, 32),
            ReLU(inplace=True),
            Linear(32, 16),
            ReLU(inplace=True),
            Linear(16, 1)
        )

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.mlp(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data =  data.to(device)

model = Net().to(device) 

torch.manual_seed(42)

lr = 1e-2
optimizer = Adam(model.parameters(), lr=lr)
epochs = 100

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.mse_loss(model()[data.train_mask],
                      data.y[data.train_mask].view(-1, 1))
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    logits = model()
    mask1 = data['train_mask']
    pred1 = logits[mask1]
    mse = nn.MSELoss()
    loss1 = torch.mean(mse(data.y[mask1].view(-1, 1), pred1)).item()
    mask = data['test_mask']
    pred = logits[mask]
    loss = torch.mean(mse(data.y[mask].view(-1, 1), pred)).item()
    return loss1, loss

train_loss = []
test_loss = []
for epoch in range(1, epochs):
  print(train())
  trainloss, testloss = test()
  train_loss.append(trainloss)
  test_loss.append(testloss)
  print(f"Epoch = {epoch}. Train Loss = {trainloss} - Test Loss = {testloss}")

dados = {"Treino": train_loss, "Teste": test_loss}
df = pd.DataFrame(data=dados)
df.plot()
plt.show()

mask = data['test_mask']
pred = model()[mask].detach().numpy()
lab = data.y[mask].view(-1, 1).detach().numpy()
plt.scatter(lab, pred)
x = np.arange(0, 18, 1)
plt.plot(x, x)
plt.show()