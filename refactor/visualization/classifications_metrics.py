import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from os.path import isdir
from os import makedirs
import pathlib
from typing import List

from refactor.utils.files import classification_metrics_file
from matplotlib.lines import Line2D

plt.rcParams["font.size"] = "12"


CLASSES = [
    "regular",
    "critical",
    "macro avg",
    "weighted avg",
]
METRICS = ["precision", "recall", "f1-score", "support"]
WIDTH = 0.08
COLORS = [
    "#f76469",
    "#828583",
    "#32a852",
]


def classification_metrics_bars(
    basedir: str,
    graphname: str,
    df: pd.DataFrame,
    bar_variable: str,
    cols_to_plot: List[str],
):
    metric_cols = ["accuracy"] + [
        f"{c}_{m}" for c, m in list(product(CLASSES, METRICS))
    ]
    parameter_cols = [c for c in list(df.columns) if c not in metric_cols]
    unique_values = {
        c: df[c].unique().tolist()
        for c in parameter_cols
        if all([c != "eval", c != bar_variable])
    }
    combinations = list(product(*[v for v in unique_values.values()]))
    for c in combinations:
        fig, ax = plt.subplots(figsize=(8, 6))
        pairs = [f"{k}{v}" for k, v in zip(list(unique_values.keys()), c)]
        sufix = "_".join(str(v) for v in pairs)
        col_filters = {
            k: df[k] == v for k, v in zip(list(unique_values.keys()), c)
        }
        df_filter = pd.DataFrame(data=col_filters).all(axis=1)
        grouped = df.loc[df_filter, :].groupby(bar_variable)
        mean = grouped.mean()
        std = grouped.std()
        indices = mean.index.tolist()
        for i, col in enumerate(cols_to_plot):
            bar_pos = np.array(indices) - WIDTH / 2.0
            ax.bar(
                bar_pos + (i + 0.5) * WIDTH / len(cols_to_plot),
                mean[col].to_numpy(),
                yerr=std[col].to_numpy(),
                capsize=5,
                color=COLORS[i],
                width=WIDTH / len(cols_to_plot),
            )
        ax.grid(axis="y", alpha=0.4)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.set_ylabel("Scores (%)")
        ax.set_xticks(indices)
        ax.set_xticklabels([str(t) for t in indices])
        ax.set_xlabel("training split")

        plt.tight_layout()
        custom_leg = [
            Line2D(
                [],
                [],
                color=COLORS[i],
                marker="s",
                linestyle="None",
                markersize=12,
            )
            for i in range(len(cols_to_plot))
        ]
        fig.legend(
            handles=custom_leg,
            labels=cols_to_plot,
            loc="lower center",
            borderaxespad=0.2,
            ncol=len(cols_to_plot),
        )
        plt.subplots_adjust(bottom=0.180)
        filename = classification_metrics_file(basedir, graphname, sufix)
        filedir = pathlib.Path(filename).parent
        if not isdir(filedir):
            makedirs(filedir)
        plt.savefig(filename)
        plt.clf()
        plt.close(fig)
