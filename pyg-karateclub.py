import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class KarateDataset(InMemoryDataset):

    def __init__(self,
                 G: nx.Graph,
                 transform=None):

        super(KarateDataset, self).__init__('.',
                                            transform,
                                            None,
                                            None)

        adj_csr: csr_matrix = nx.to_scipy_sparse_matrix(G)
        adj: coo_matrix = adj_csr.tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        # Edge index é um tensor do torch com shape 2 x 156, que contém os
        # pares (s, d) dos nós para definição das arestas.
        # Contém 2*m arestas para grafos simples, pois conta nas duas
        # direções.

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
        labels = np.asarray([G.nodes[i]["club"] != "Mr. Hi" for i in G.nodes]).astype(np.int64)
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        data.num_classes = 2

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


G = nx.karate_club_graph()
dataset = KarateDataset(G)
data = dataset[0]
data.num_classes
data["train_mask"]

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
        self.conv2 = GCNConv(16, int(data.num_classes))

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data =  data.to(device)

model = Net().to(device) 

torch.manual_seed(42)

lr = 1e-3
optimizer = Adam(model.parameters(), lr=lr)
epochs = 5000

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    logits = model()
    mask1 = data['train_mask']
    pred1 = logits[mask1].max(1)[1]
    acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
    mask = data['test_mask']
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc1,acc

train_loss = []
test_loss = []
for epoch in range(1, epochs):
  train()
  trainloss, testloss = test()
  train_loss.append(trainloss)
  test_loss.append(testloss)

dados = {"Treino": train_loss, "Teste": test_loss}
df = pd.DataFrame(data=dados)
df.plot()
plt.show()

train_acc,test_acc = test()

print('#' * 70)
print('Train Accuracy: %s' % train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)
