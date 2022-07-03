import warnings
import pandas as pd
import numpy as np
from os import curdir
from os.path import join
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from refactor.utils.files import embeddings_result_file

GRAPHNAME = "ieee118"
RESULT_BASEDIR = join(curdir, "results", "clg")
FIGURE_BASEDIR = join(RESULT_BASEDIR, "figures")


def _generate_embeddings_to_plot(df: pd.DataFrame) -> pd.DataFrame:
    embedding_cols = [c for c in df.columns if "z" in c]
    N_COMPONENTS = 2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        z = TSNE(
            n_components=N_COMPONENTS,
            learning_rate="auto",
            init="pca",
            random_state=0,
        ).fit_transform(df[embedding_cols].to_numpy())
    df_plot = pd.DataFrame(
        z, columns=[f"z{i}" for i in range(1, N_COMPONENTS + 1)]
    )
    df_plot["label"] = df["label"].tolist()
    return df_plot.loc[df_plot["label"] >= 0]


df = pd.read_csv(
    embeddings_result_file(RESULT_BASEDIR, GRAPHNAME), index_col=0
)

COLORS = np.array(
    [
        [0.5098039215686274, 0.5215686274509804, 0.5137254901960784, 1],
        [0.9686274509803922, 0.39215686274509803, 0.4117647058823529, 1],
    ]
)
cm = ListedColormap(COLORS)
legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        lw=0,
        label="Regular",
        markerfacecolor=cm(0),
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        lw=0,
        label="Critical",
        markerfacecolor=cm(1),
        markersize=10,
    ),
]

tol = 0.5
train_split = 0.5
eval = 30

embeddings_cols = [c for c in df.columns if "z" in c]
common_filter = (
    (df["tol"] == tol)
    & (df["train_split"] == train_split)
    & (df["eval"] == 30)
)

for epoch in [1, 300]:
    df_epoch = df.loc[
        common_filter & (df["epoch"] == epoch), embeddings_cols + ["label"]
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    df_plot = _generate_embeddings_to_plot(df_epoch)
    df_plot.plot.scatter(
        x="z1",
        y="z2",
        c="label",
        colormap=cm,
        s=50,
        alpha=0.8,
        ax=ax,
        colorbar=None,
    )
    ax.legend(handles=legend, loc="upper left")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(join(curdir, "sbpo2022", f"embedding_evolution{epoch}.eps"))
    plt.clf()
    plt.close(fig)
