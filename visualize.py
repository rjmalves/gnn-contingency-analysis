import sys
import pandas as pd
from os import getenv
from os.path import join
from dotenv import load_dotenv

from refactor.utils.files import (
    train_result_file,
    roc_result_file,
    class_result_file,
)
from refactor.visualization.training import stacked_train_curve
from refactor.visualization.roc_auc import stacked_roc_curve
from refactor.visualization.classifications_metrics import (
    classification_metrics_bars,
)


if __name__ == "__main__":
    load_dotenv(override=True)

    if len(sys.argv) == 1:
        raise ValueError("Please specify a case")

    # Study case parameters
    GRAPHNAME = getenv("GRAPHNAME")
    EDGELIST_BASEDIR = getenv("EDGELIST_BASEDIR")
    CRITICALITY_BASEDIR = getenv("CRITICALITY_BASEDIR")
    RESULT_BASEDIR = join(getenv("RESULT_BASEDIR"), sys.argv[1])
    FIGURE_BASEDIR = join(RESULT_BASEDIR, "figures")

    train_result = pd.read_csv(
        train_result_file(RESULT_BASEDIR, GRAPHNAME), index_col=0
    )
    stacked_train_curve(FIGURE_BASEDIR, GRAPHNAME, train_result)

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
