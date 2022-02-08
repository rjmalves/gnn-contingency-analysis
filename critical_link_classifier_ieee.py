import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import Node2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from contingency.models.network import Network
from contingency.controllers.screener import ExhaustiveScreener

# Lê o arquivo do grafo da lista de arestas
G = nx.read_edgelist("data/ieee39.txt")

# Cria as labels das arestas analisando contingências
k = 1
net = Network("IEEE", G)
screener = ExhaustiveScreener(net)
deltas = screener.normalized_global_deltas(k)
max_delta = max(list(deltas.values()))
for d in deltas.keys():
    if deltas[d]/max_delta >= 0.7:
        deltas[d] = 1
    else:
        deltas[d] = 0
nx.set_edge_attributes(G, deltas, "label")

pos = nx.spring_layout(G)
nx.draw(G,pos)
nx.draw_networkx_edge_labels(G,pos,nx.get_edge_attributes(G,'label'))
plt.show()


# Converte para Pytorch Geometric
data = from_networkx(G, group_edge_attrs=["label"])

# Extrai os nós do grafo

np.random.seed(10)

nodes = data.edge_index.t().numpy()
nodes = np.unique(list(nodes[:,0]) + list(nodes[:,1]))

np.random.shuffle(nodes) 
print(len(nodes))


# get train test and val sizes: (70% - 15% - 15%)
train_size = int(len(nodes)*0.7)
test_size = int(len(nodes)*0.85) - train_size
val_size = len(nodes) - train_size - test_size

# get train test and validation set of nodes
train_set = nodes[0:train_size]
test_set = nodes[train_size:train_size+test_size]
val_set = nodes[train_size+test_size:]

print(len(train_set),len(test_set),len(val_set))
print(len(train_set)+len(test_set)+len(val_set) == len(nodes))

print("train set\t",train_set[:10])
print("test set \t",test_set[:10])
print("val set  \t",val_set[:10])


# build test train val masks
device = "cpu"
train_mask = torch.zeros(len(nodes),dtype=torch.long, device=device)
for i in train_set:
    train_mask[i] = 1.

test_mask = torch.zeros(len(nodes),dtype=torch.long, device=device)
for i in test_set:
    test_mask[i] = 1.
    
val_mask = torch.zeros(len(nodes),dtype=torch.long, device=device)
for i in val_set:
    val_mask[i] = 1.
    
print("train mask \t",train_mask[0:15])
print("test mask  \t",test_mask[0:15])
print("val mask   \t",val_mask[0:15]) 

# remove from the data what do we not use.

print("befor\t\t",data)
data.x = None
data.edge_attr = None
data.y = None

# add masks
data.train_mask = train_mask
data.test_mask = test_mask
data.val_mask = val_mask


print("after\t\t",data)


model = Node2Vec(data.edge_index, embedding_dim=16, walk_length=10,
                 context_size=10, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=32, shuffle=True, num_workers=4)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=10)
    return acc


for epoch in range(1, 51):
    loss = train()
    #acc = test()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

model.embedding

z = model()


# from tensor to numpy
emb_128 = z.detach().cpu().numpy()


# fit and transform using PCA
pca = PCA(n_components=2)
emb2d = pca.fit_transform(emb_128)


plt.title("node embedding in 2D")
plt.scatter(emb2d[:,0],emb2d[:,1])
plt.show()


data = from_networkx(G, group_edge_attrs=["label"])
data.edge_attr
data


# convert edge attributes from categorical to numerical
edge_attr_cat = data.edge_attr.numpy()
print("Categorical edge attributes:\n",edge_attr_cat[:3])

edge_attr = np.squeeze(data.edge_attr.numpy())

print("\n\nNumerical edge attributes:\n",edge_attr[:3])

# compute edge embedding

edge_embedding = []
for u,v in data.edge_index.t():
    edge_embedding.append(np.mean([emb_128[u],emb_128[v]],0))

# fit and transform using PCA
pca = PCA(n_components=2)
edge_emb2d = pca.fit_transform(edge_embedding)
df = pd.DataFrame(dict(edge_att=edge_attr))
colors = {0:"red",1:"blue"}
plt.title("edge embedding in 2D")
plt.scatter(edge_emb2d[:,0],edge_emb2d[:,1],c=df.edge_att.map(colors))
plt.show()


# not so good but we are using PCA to reduce the dim from 128 to 2

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=10)

scores = cross_val_score(clf,
                         edge_embedding,
                         edge_attr,
                         cv=4,
                         scoring=make_scorer(recall_score))
np.mean(scores)
scores
