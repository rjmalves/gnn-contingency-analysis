import random
import sys
from os import getenv
from os.path import join

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from refactor.utils.files import (
    class_result_file,
    embeddings_result_file,
    roc_result_file,
    train_result_file,
)
from refactor.visualization.classifications_metrics import (
    classification_metrics_bars,
)
from refactor.visualization.embeddings import training_embeddings_scatter
from refactor.visualization.roc_auc import stacked_roc_curve
from refactor.visualization.training import stacked_train_curve

random.seed(0)
np.random.seed(0)


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

    embeddings_result = pd.read_csv(
        embeddings_result_file(RESULT_BASEDIR, GRAPHNAME), index_col=0
    )
    training_embeddings_scatter(FIGURE_BASEDIR, GRAPHNAME, embeddings_result)
