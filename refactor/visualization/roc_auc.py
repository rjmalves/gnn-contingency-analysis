import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from os.path import isdir
from os import makedirs
import pathlib

from refactor.utils.files import roc_curve_file

plt.rcParams["font.size"] = "12"


def stacked_roc_curve(basedir: str, graphname: str, df: pd.DataFrame):
    parameter_cols = [
        c for c in list(df.columns) if c not in ["fpr", "tpr", "threshold"]
    ]
    unique_values = {c: df[c].unique().tolist() for c in parameter_cols}
    combinations = list(product(*[v for v in unique_values.values()]))
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in combinations:
        pairs = [
            f"{k}{v}"
            for k, v in zip(list(unique_values.keys()), c)
            if k != "eval"
        ]
        sufix = "_".join(str(v) for v in pairs)
        col_filters = {
            k: df[k] == v for k, v in zip(list(unique_values.keys()), c)
        }
        df_filter = pd.DataFrame(data=col_filters).all(axis=1)
        ax.plot(
            df.loc[df_filter, "fpr"].to_numpy(),
            df.loc[df_filter, "tpr"].to_numpy(),
            alpha=0.2,
            color="red",
        )
    ax.plot([0, 1], [0, 1], linestyle="dashed", color="grey", alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    filename = roc_curve_file(basedir, graphname, sufix)
    filedir = pathlib.Path(filename).parent
    if not isdir(filedir):
        makedirs(filedir)
    plt.savefig(filename)
    plt.clf()
    plt.close(fig)
