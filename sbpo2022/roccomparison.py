import warnings
import pandas as pd
import numpy as np
from os import curdir
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from refactor.utils.files import roc_result_file

GRAPHNAME = "ieee300"
RESULT_BASEDIR = join(curdir, "results", "clg")
FIGURE_BASEDIR = join(RESULT_BASEDIR, "figures")


df = pd.read_csv(roc_result_file(RESULT_BASEDIR, GRAPHNAME), index_col=0)

COLORS = (
    np.array(
        [
            [247, 100, 105, 255],
            [49, 85, 214, 255],
            [50, 158, 82, 255],
            [130, 133, 131, 255],
        ]
    )
    / 255.0
)
cm = ListedColormap(COLORS)


STYLES = ["solid", "dashed", "dotted"]

legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        lw=2,
        label="t = 10%",
        color=COLORS[0],
        markersize=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        lw=1,
        label="p = 90",
        color="black",
        markersize=0,
        linestyle="solid",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        lw=2,
        label="t = 30%",
        color=COLORS[1],
        markersize=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        lw=1,
        label="p = 75",
        color="black",
        markersize=0,
        linestyle="dashed",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        lw=2,
        label="t = 50%",
        color=COLORS[2],
        markersize=0,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        lw=1,
        label="p = 50",
        color="black",
        markersize=0,
        linestyle="dotted",
    ),
]


# IEEE 118
# TOLS = [0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5]
# TLS = [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5]
# EVALS = [29, 21, 30, 30, 30, 30, 30, 30, 30]

# IEEE 300
# TOLS = [0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5]
# TLS = [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5]
# EVALS = [10, 21, 30, 17, 1, 1, 26, 11, 26]

# ----- RED ----- | ----- BLU ----- | ----- GRN ----- |
# SOL | DAS | DOT | SOL | DAS | DOT | SOL | DAS | DOT |
TOLS = [0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5]
TLS = [0.1, 0.3, 0.5, 0.1, 0.3, 0.5, 0.1, 0.3, 0.5]
EVALS = [10, 21, 30, 17, 1, 1, 26, 11, 26]

fig, ax = plt.subplots(figsize=(5, 5))
for i, (tol, train_split, eval) in enumerate(zip(TOLS, TLS, EVALS)):
    color = COLORS[i // 3]
    style = STYLES[i % 3]
    common_filter = (
        (df["quantile"] == tol)
        & (df["train_split"] == train_split)
        & (df["eval"] == eval)
    )
    df_plot = df.loc[common_filter, ["fpr", "tpr"]]
    ax.plot(
        df_plot["fpr"].to_numpy(),
        df_plot["tpr"].to_numpy(),
        color=color,
        linewidth=1.5,
        linestyle=style,
    )
ax.plot([0, 1], [0, 1], linestyle="dashed", color="grey", alpha=0.5)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.legend(handles=legend, ncol=3)
plt.tight_layout()
plt.savefig(join(curdir, "sbpo2022", f"roccomparison_{GRAPHNAME}.eps"))
plt.clf()
plt.close(fig)
