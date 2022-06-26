import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = "12"

fig, ax = plt.subplots(figsize=(5, 5))

ax.stairs(
    preds["tpr"],
    np.concatenate((np.array([0.0]), preds["fpr"])),
    alpha=0.2,
    color="red",
    baseline=None,
)
ax.plot([0, 1], [0, 1], linestyle="dashed", color="grey", alpha=0.5)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
plt.tight_layout()
plt.savefig(f"./figures/{GRAPH}_{k}_{tol}_{train_split}_{channels}.png")
plt.clf()
