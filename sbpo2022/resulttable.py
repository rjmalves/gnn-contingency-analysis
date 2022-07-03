import pandas as pd
import numpy as np
from os import curdir
from os.path import join

from refactor.utils.files import class_result_file, auc_result_file


RESULT_BASEDIR = join(curdir, "results", "clg")
GRAPHS = ["ieee39", "ieee57", "ieee118", "ieee300"]

dfs_class = {
    g: pd.read_csv(class_result_file(RESULT_BASEDIR, g), index_col=0)
    for g in GRAPHS
}

dfs_auc = {
    g: pd.read_csv(auc_result_file(RESULT_BASEDIR, g), index_col=0)
    for g in GRAPHS
}

for g in GRAPHS:
    dfs_class[g]["graph"] = g
    dfs_auc[g]["graph"] = g


df_class_final = pd.DataFrame()
df_auc_final = pd.DataFrame()
for g in GRAPHS:
    df_class_final = pd.concat(
        [df_class_final, dfs_class[g]], ignore_index=True
    )
    df_auc_final = pd.concat([df_auc_final, dfs_auc[g]], ignore_index=True)

df_class_final.groupby(["graph", "train_split", "quantile"])[
    "critical_f1-score"
].mean()

df_auc_final.groupby(["graph", "train_split", "quantile"])["auc"].mean()
