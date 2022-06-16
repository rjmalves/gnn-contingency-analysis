import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

arq = "a3_ieee300"
metric_cols = ["mse", "mae", "r2"]
legends = ["MSE", "MAE", "R2"]
colors = [
    "#f76469",
    "#828583",
    "#32a852",
]
split_col = "train_split"
splits = [0.5]
k_col = "k"
width = 0.6
df = pd.read_csv(f"./result_{arq}.csv", index_col=0)
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)


# Edges by split table
# 0.1: [1, 1, 1, 1]
# 0.2: [1, 1, 1, 1]
# 0.3: [1, 1, 2, 2]
# 0.4: [1, 1, 2, 2]
# 0.5: [2, 2, 3, 2]

# Used for A1 Approach
# splits_by_k = {1: [0.3, 0.5], 2: [0.2, 0.5], 3: [0.1, 0.4, 0.5], 4: [0.1, 0.5]}
# labels_by_k = {
#     1: [1, 2],
#     2: [1, 2],
#     3: [1, 2, 3],
#     4: [1, 2],
# }

# Used for A2 approach
# splits_by_k = {1: [0.1, 0.5], 2: [0.4, 0.5], 3: [0.2, 0.4, 0.5], 4: [0.2, 0.5]}
# labels_by_k = {
#     1: [1, 2],
#     2: [1, 2],
#     3: [1, 2, 3],
#     4: [1, 2],
# }

# Used for A3 Approach
ks = [1, 2, 3]
splits = [0.7]
splits_by_k = {k: splits for k in ks}
labels_by_k = {
    1: splits,
    2: splits,
    3: splits,
}


xticks = splits
for i, k in enumerate([1, 2, 3]):
    ix = i // 2
    iy = i % 2

    print(f"k = {k}")

    for j, col in enumerate(metric_cols):
        values = [
            df.loc[(df[k_col] == k) & (df[split_col] == s), col]
            for s in splits_by_k[k]
        ]
        means = [v.mean() for v in values]
        errors = [v.std() for v in values]
        bar_pos = np.array(labels_by_k[k]) - width / 2.0
        axs[ix, iy].bar(
            bar_pos + (j + 0.5) * width / len(metric_cols),
            means,
            yerr=errors,
            capsize=5,
            color=colors[j],
            width=width / len(metric_cols),
        )
    axs[ix, iy].set_title(f"k = {k}")

axs[ix, iy].set_xticks(xticks)
axs[ix, iy].set_xticklabels([str(lab) for lab in xticks])

# axs[0, 0].set_ylim(0, 1.0)
axs[0, 0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axs[0, 0].set_yticklabels(["0", "25", "50", "75", "100"])
axs[0, 0].set_ylabel("Scores")
axs[1, 0].set_ylabel("Scores")
axs[1, 0].set_xlabel("# of critical edges in training")
axs[1, 1].set_xlabel("# of critical edges in training")
plt.tight_layout()
custom_leg = [
    Line2D([], [], color=colors[i], marker="s", linestyle="None", markersize=12)
    for i in range(len(legends))
]
fig.legend(
    handles=custom_leg,
    labels=legends,
    loc="lower center",
    borderaxespad=0.2,
    ncol=len(legends),
)
plt.subplots_adjust(bottom=0.140)
plt.savefig(f"visual_{arq}_metrics.png")

for i, k in enumerate([1, 2, 3]):
    labels = labels_by_k[k]
    lines = [[str(int(100 * lab)) + "\% "] for lab in labels]
    for j, col in enumerate(metric_cols):
        values = [
            df.loc[(df[k_col] == k) & (df[split_col] == s), col]
            for s in splits_by_k[k]
        ]
        means = [v.mean() for v in values]
        errors = [v.std() for v in values]
        for m in range(len(lines)):
            lines[m].append(f" ${means[m]:.1f} \\pm {errors[m]:.1f}$ ")
    print(f"k = {k}")
    for line in lines:
        print("&".join(line) + " \\\\")
