from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from itertools import product
from os.path import isdir
from os import makedirs
import pathlib
import warnings

from refactor.utils.files import embedding_scatter_file

plt.rcParams["font.size"] = "10"


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


def training_embeddings_scatter(
    basedir: str, graphname: str, df: pd.DataFrame
):

    # COLORS = [
    #     "#828583",
    #     "#f76469",
    #     "#32a852",
    # ]

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

    embedding_cols = [c for c in df.columns if "z" in c]

    parameter_cols = [
        c for c in list(df.columns) if c not in embedding_cols + ["label"]
    ]
    unique_values = {
        c: df[c].unique().tolist() for c in parameter_cols if c != "epoch"
    }
    epochs = df["epoch"].unique().tolist()
    epochs_plot = [min(epochs), max(epochs)]
    combinations = list(product(*[v for v in unique_values.values()]))
    for c in combinations:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        pairs = [f"{k}{v}" for k, v in zip(list(unique_values.keys()), c)]
        sufix = "_".join(str(v) for v in pairs)
        col_filters = {
            k: df[k] == v for k, v in zip(list(unique_values.keys()), c)
        }
        for i, epoch in enumerate(epochs_plot):
            col_filters["epoch"] = df["epoch"] == epoch
            df_filter = pd.DataFrame(data=col_filters).all(axis=1)
            df_plot = _generate_embeddings_to_plot(df.loc[df_filter, :])
            df_plot.plot.scatter(
                x="z1",
                y="z2",
                c="label",
                colormap=cm,
                s=50,
                alpha=0.8,
                ax=axs[i],
                colorbar=None,
            )
            axs[i].set_title(f"epoch = {epoch}")
            axs[i].legend(handles=legend, loc="upper right")
            axs[i].set_ylabel("")
            axs[i].set_xlabel("")
            axs[i].set_yticks([])
            axs[i].set_xticks([])
        plt.tight_layout()
        filename = embedding_scatter_file(basedir, graphname, sufix)
        filedir = pathlib.Path(filename).parent
        if not isdir(filedir):
            makedirs(filedir)
        plt.savefig(filename)
        plt.clf()
        plt.close(fig)
