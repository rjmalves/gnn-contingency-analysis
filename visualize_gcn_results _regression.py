import pandas as pd
import matplotlib.pyplot as plt

arq = "k1_regression_arch1"
df = pd.read_csv(f"./result_{arq}.csv", index_col=0)
fig, axs = plt.subplots(3, 1, figsize=(28, 20), sharex=True)
col_group = ["embedding_d", "channels", "train_split"]
df.boxplot(
    column=["mse"],
    by=col_group,
    rot=90,
    ax=axs[0],
)
df.boxplot(
    column=["mae"],
    by=col_group,
    rot=90,
    ax=axs[1],
)
df.boxplot(
    column=["r2"],
    by=col_group,
    rot=90,
    ax=axs[2],
)
axs[0].set_ylim(0, 0.3)
axs[1].set_ylim(0, 0.4)
axs[2].set_ylim(-1, 1)
plt.tight_layout()
plt.savefig(f"visual_{arq}.png")
