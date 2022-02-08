import numpy as np

from ge.classify import Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx


def load_labels():
    G = nx.karate_club_graph()
    labels = nx.get_node_attributes(G, "club")
    classes = list(set(labels.values()))
    X = [str(k) for k in labels.keys()]
    Y = [str(classes.index(v)) for k, v in labels.items()]
    return X, Y


def evaluate_embeddings(embeddings):
    X, Y = load_labels()
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(
    embeddings,
):
    X, Y = load_labels()
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

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
    plt.savefig("karate_deepwalk.svg")
    plt.clf()


if __name__ == "__main__":
    G = nx.karate_club_graph()
    labels = nx.get_node_attributes(G, "club")
    classes = list(set(labels.values()))
    G = nx.read_edgelist(
        "./karate.edgelist",
        create_using=nx.Graph(),
        nodetype=None,
        data=[("weight", int)],
    )

    model = DeepWalk(G, walk_length=5, num_walks=20, workers=1)
    res = model.train(embed_size=2, window_size=5, iter=50)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)
