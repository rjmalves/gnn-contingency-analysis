import pandas as pd
import matplotlib.pyplot as plt

arq = "k1_ieee57_lr01_2class"
df = pd.read_csv(f"./result_{arq}.csv", index_col=0)
df = df[df["embedding_d"] == 128]
fig, axs = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
col_group = ["tol_inf", "tol_sup", "train_split", "dropout"]
df.boxplot(
    column=["accuracy"],
    by=col_group,
    rot=90,
    ax=axs[0],
)
df.boxplot(
    column=["macro avg_f1-score"],
    by=col_group,
    rot=90,
    ax=axs[1],
)
df.boxplot(
    column=["weighted avg_f1-score"],
    by=col_group,
    rot=90,
    ax=axs[2],
)
df.boxplot(
    column=["critical_f1-score"],
    by=col_group,
    rot=90,
    ax=axs[3],
)
df.boxplot(
    column=["regular_f1-score"],
    by=col_group,
    rot=90,
    ax=axs[4],
)
plt.tight_layout()
plt.savefig(f"visual_{arq}_emb128.png")
