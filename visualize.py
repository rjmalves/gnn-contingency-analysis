import pandas as pd
from os import getenv
from dotenv import load_dotenv

from refactor.utils.files import roc_result_file, class_result_file
from refactor.visualization.roc_auc import stacked_roc_curve
from refactor.visualization.classifications_metrics import (
    classification_metrics_bars,
)

load_dotenv(override=True)

# Study case parameters
GRAPHNAME = getenv("GRAPHNAME")
EDGELIST_BASEDIR = getenv("EDGELIST_BASEDIR")
CRITICALITY_BASEDIR = getenv("CRITICALITY_BASEDIR")
RESULT_BASEDIR = getenv("RESULT_BASEDIR")
FIGURE_BASEDIR = getenv("FIGURE_BASEDIR")


class_result = pd.read_csv(
    class_result_file(RESULT_BASEDIR, GRAPHNAME), index_col=0
)
classification_metrics_bars(
    FIGURE_BASEDIR,
    GRAPHNAME,
    class_result,
    bar_variable="train_split",
    cols_to_plot=["macro avg_f1-score", "critical_f1-score"],
)

roc_result = pd.read_csv(
    roc_result_file(RESULT_BASEDIR, GRAPHNAME), index_col=0
)
stacked_roc_curve(FIGURE_BASEDIR, GRAPHNAME, roc_result)
