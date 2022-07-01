import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from os.path import isdir
from os import makedirs
import pathlib

from refactor.utils.files import train_curve_file

plt.rcParams["font.size"] = "12"


def stacked_train_curve(basedir: str, graphname: str, df: pd.DataFrame):
    parameter_cols = [
        c
        for c in list(df.columns)
        if c not in ["epoch", "train_loss", "val_loss"]
    ]
    unique_values = {
        c: df[c].unique().tolist() for c in parameter_cols if c != "eval"
    }
    unique_evals = df["eval"].unique().tolist()
    combinations = list(product(*[v for v in unique_values.values()]))
    for c in combinations:
        fig, ax = plt.subplots(figsize=(5, 5))
        pairs = [f"{k}{v}" for k, v in zip(list(unique_values.keys()), c)]
        sufix = "_".join(str(v) for v in pairs)
        col_filters = {
            k: df[k] == v for k, v in zip(list(unique_values.keys()), c)
        }
        df_filter = pd.DataFrame(data=col_filters).all(axis=1)
        for ev in unique_evals:
            df_plot = df.loc[df_filter, :]
            df_plot = df_plot.loc[df_plot["eval"] == ev, :]
            ax.plot(
                df_plot["epoch"].to_numpy().flatten(),
                df_plot["train_loss"].to_numpy().flatten(),
                alpha=0.2,
                color="blue",
                label="train",
            )
            ax.plot(
                df_plot["epoch"].to_numpy().flatten(),
                df_plot["val_loss"].to_numpy().flatten(),
                alpha=0.2,
                color="red",
                label="validation",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        plt.tight_layout()
        filename = train_curve_file(basedir, graphname, sufix)
        filedir = pathlib.Path(filename).parent
        if not isdir(filedir):
            makedirs(filedir)
        plt.savefig(filename)
        plt.clf()
        plt.close(fig)
