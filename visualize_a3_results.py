import pandas as pd
import matplotlib.pyplot as plt

arq = "a3_itaipu11"
df = pd.read_csv(f"./result_{arq}.csv", index_col=0)
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
col_group = ["k", "embedding_d", "channels", "train_split"]
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
plt.tight_layout()
plt.savefig(f"visual_{arq}.png")
