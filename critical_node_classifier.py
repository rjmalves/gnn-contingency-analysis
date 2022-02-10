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


g = nx.path_graph(7)
nx.current_flow_betweenness_centrality(g)

# Lê os grafos do arquivo .g6
Gs = nx.read_graph6("data/itaipu11_random.g6")
print(f"Lidos {len(Gs)} grafos")
Gs = [G for G in Gs if nx.is_connected(G)]
print(f"Resumidos em {len(Gs)} grafos conexos")

# Para cada grafo lido, adiciona as labels das arestas
k = 1
TRAIN_SPLIT = 0.7
NUM_EPOCHS_TRAIN = 50

Gs = nx.read_edgelist("./data/ieee39.txt")


def compara_delta_cfb_ciclos(ni: int, nf: int, arestas: list):
    max_deltas = []
    nos = list(range(ni, nf + 1))
    for i in nos:
        G = nx.cycle_graph(i)
        for a in arestas:
            G.add_edge(a)
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
        net = Network("Cycle", G)
        k = 1
        screener = ExhaustiveScreener(net)
        deltas = screener.global_deltas(k)
        max_deltas.append(max(list(deltas.values())))
    return nos, max_deltas

    plt.scatter(nos, max_deltas)
    plt.show()


print("Gerando labels das arestas")
k = 1
for G in Gs:
    net = Network("IEEE", Gs)
    screener = ExhaustiveScreener(net)
    deltas = screener.global_deltas(k)
    print(deltas)
    max_delta = max(list(deltas.values()))
    print(max_delta)
    for d in deltas.keys():
        if deltas[d] / max_delta >= 0.7:
            deltas[d] = 1
        else:
            deltas[d] = 0
    nx.set_edge_attributes(Gs, deltas, "label")

Gl = nx.line_graph(Gs)
nx.set_node_attributes(Gl, deltas, "label")

x = np.arange(3.0, 100.0, 1.0)
y_cycle = np.zeros_like(x)
y_path = np.zeros_like(x)
for i, xi in enumerate(x):
    if xi % 2 == 0:
        y_path[i] = xi * (xi - 2) / (2 * xi * (xi - 1))
        y_cycle[i] = (xi - 2) ** 2 / (4 * xi * (xi - 1))
    else:
        y_path[i] = (xi - 1) ** 2 / (2 * xi * (xi - 1))
        y_cycle[i] = (xi - 1) * (xi - 3) / (4 * xi * (xi - 1))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y_cycle, label="cycle")
ax.plot(x, y_path, label="path")

plt.show()


# pos = nx.spring_layout(Gs[0])
# nx.draw(Gs[0],pos)
# nx.draw_networkx_edge_labels(Gs[0],pos,nx.get_edge_attributes(Gs[0],'label'))
# plt.show()

np.random.seed(2021)

X = []
Y = []

for G in Gs:

    # Converte para Pytorch Geometric
    data = from_networkx(G, group_edge_attrs=["label"])

    # Extrai os nós do grafo
    nodes = data.edge_index.t().numpy()
    nodes = np.unique(list(nodes[:, 0]) + list(nodes[:, 1]))

    np.random.shuffle(nodes)

    # get train test and val sizes: (70% - 15% - 15%)
    train_size = int(len(nodes) * TRAIN_SPLIT)
    test_size = int(len(nodes)) - train_size

    # get train test and validation set of nodes
    train_set = nodes[0:train_size]
    test_set = nodes[train_size : train_size + test_size]

    # build test train val masks
    device = "cpu"
    train_mask = torch.zeros(len(nodes), dtype=torch.long, device=device)
    for i in train_set:
        train_mask[i] = 1.0

    test_mask = torch.zeros(len(nodes), dtype=torch.long, device=device)
    for i in test_set:
        test_mask[i] = 1.0

    # remove from the data what do we not use.
    data.x = None
    data.edge_attr = None
    data.y = None

    # add masks
    data.train_mask = train_mask
    data.test_mask = test_mask

    model = Node2Vec(
        data.edge_index,
        embedding_dim=16,
        walk_length=5,
        context_size=5,
        walks_per_node=5,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=8, shuffle=True, num_workers=4)
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
        acc = model.test(
            z[data.train_mask],
            data.y[data.train_mask],
            z[data.test_mask],
            data.y[data.test_mask],
            max_iter=10,
        )
        return acc

    for epoch in range(1, NUM_EPOCHS_TRAIN + 1):
        loss = train()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")

    z = model()

    # from tensor to numpy
    emb_128 = z.detach().cpu().numpy()

    data = from_networkx(G, group_edge_attrs=["label"])
    edge_attr = np.squeeze(data.edge_attr.numpy())
    edge_embedding = []
    for u, v in data.edge_index.t():
        edge_embedding.append(np.mean([emb_128[u], emb_128[v]], 0))

    X += edge_embedding
    Y += list(edge_attr)

X = np.stack(X)
Y = np.array(Y)
df = pd.DataFrame(X)
df.columns = [f"X{i}" for i in range(X.shape[1])]
df["Y"] = Y
df.to_csv("dados.csv")

df = pd.read_csv("dados_itaipu11.csv", index_col=0)
cols_x = [c for c in list(df.columns) if "X" in c]
X = df[cols_x].to_numpy()
Y = df["Y"].to_numpy()

from sklearn.model_selection import cross_val_score
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=10)

print(f"% de 0: {100 * len(Y[Y == 0]) / len(Y)}")
print(f"% de 1: {100 * len(Y[Y == 1]) / len(Y)}")

scores = cross_val_score(clf, X, Y, cv=4, scoring=make_scorer(f1_score))
np.mean(scores)
scores

clf.fit(X, Y)
yaprox = clf.predict(X)
