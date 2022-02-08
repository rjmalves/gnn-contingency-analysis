"""
Run the graph embedding methods on Karate graph and evaluate them on 
graph reconstruction and visualization. Please copy the 
gem/data/karate.edgelist to the working directory
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gem.utils import graph_util

from gem.embedding.lap import LaplacianEigenmaps


def load_labels():
    G = nx.karate_club_graph()
    labels = nx.get_node_attributes(G, "club")
    classes = list(set(labels.values()))
    X = [str(k) for k in labels.keys()]
    Y = [str(classes.index(v)) for k, v in labels.items()]
    return X, Y


def plot_embeddings(emb_list):
    X, Y = load_labels()

    node_labels = [str(i + 1) for i in range(len(emb_list))]
    class_labels = {"0": "Mr. Hi", "1": "Officer"}
    class_colors = {"0": "#f76468", "1": "#524b4a"}
    node_classes = {c: [] for c in class_labels.keys()}
    node_positions = {c: [] for c in class_labels.keys()}
    for i in range(len(Y)):
        node_classes[Y[i]].append(node_labels[i])
        node_positions[Y[i]].append(list(emb_list[i, :]))
    for n in class_labels.keys():
        node_positions[n] = np.array(node_positions[n])

    fig, ax = plt.subplots(figsize=(5, 5))

    for c, l in class_labels.items():
        ax.scatter(
            node_positions[c][:, 0],
            node_positions[c][:, 1],
            label=l,
            s=180,
            alpha=0.5,
            c=class_colors[c],
        )
        for i in range(node_positions[c].shape[0]):
            ax.annotate(
                node_classes[c][i],
                (node_positions[c][i, 0], node_positions[c][i, 1]),
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.legend()
    plt.tight_layout()
    plt.savefig("karate_le.svg")
    plt.clf()


if __name__ == "__main__":
    edge_f = "./karate.edgelist"
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    model = LaplacianEigenmaps(d=2)
    Y, t = model.learn_embedding(
        graph=G, edge_f=None, is_weighted=False, no_python=True
    )
    plot_embeddings(Y)
