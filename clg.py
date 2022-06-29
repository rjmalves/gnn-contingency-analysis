import networkx as nx
import pandas as pd
from itertools import product
import torch
from os import getenv, makedirs
from os.path import isdir, join
from dotenv import load_dotenv
from typing import List


from refactor.approaches.labeling import QuantileLabeling
from refactor.approaches.postprocessing import Postprocessing
from refactor.approaches.preprocessing import Preprocessing
from refactor.approaches.clg import CLG, train, test
from refactor.utils.files import (
    edgelist_file,
    criticality_file,
    class_result_file,
    auc_result_file,
    roc_result_file,
    train_result_file,
)

load_dotenv(override=True)

# Study case parameters
GRAPHNAME = getenv("GRAPHNAME")
EDGELIST_BASEDIR = getenv("EDGELIST_BASEDIR")
CRITICALITY_BASEDIR = getenv("CRITICALITY_BASEDIR")
RESULT_BASEDIR = join(getenv("RESULT_BASEDIR"), "clg")
FIGURE_BASEDIR = join(RESULT_BASEDIR, "figures")
K = [int(k) for k in getenv("K").split(",") if len(k) > 0]
EDGELIST = edgelist_file(EDGELIST_BASEDIR, GRAPHNAME)
TOL = [0.25]
N_EVALS = 50

# Training parameters
TRAIN_SPLIT = [0.5]
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3

# Model parameters
EMBEDDING_D = 128
DROPOUT = 0.5
HIDDEN_CHANNELS = 16


G = nx.read_edgelist(EDGELIST)
combinations = list(product(K, TOL, TRAIN_SPLIT))

train_result = pd.DataFrame()
class_result = pd.DataFrame()
auc_result = pd.DataFrame()
roc_result = pd.DataFrame()
for c in combinations:
    k, tol, train_split = c
    CRITICALITY = criticality_file(CRITICALITY_BASEDIR, GRAPHNAME, k)
    labeling_strategy = QuantileLabeling(tol)
    preprocessor = Preprocessing(
        G,
        CRITICALITY,
        train_split=train_split,
        embedding_dimension=EMBEDDING_D,
        labeling_strategy=labeling_strategy,
    )
    print(f"Params = {c}")
    for i in range(1, N_EVALS + 1):
        print(f"Eval {i}")
        name = f"k{k}_split{train_split}_it{i}"
        data = preprocessor.torch_node_data

        model = CLG(
            num_inputs=data.num_features,
            hidden_channels=HIDDEN_CHANNELS,
            num_outputs=data.num_classes,
            dropout=DROPOUT,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss()
        train_losses: List[float] = []
        val_losses: List[float] = []
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, val_loss = train(model, data, optimizer, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        y, yhat = test(model, data)
        postprocessor = Postprocessing(
            y,
            yhat,
            train_losses,
            val_losses,
            k,
            train_split,
            labeling_strategy,
            eval=i,
        )
        train_result = Postprocessing.update_report(
            train_result, postprocessor.train_report()
        )
        class_result = Postprocessing.update_report(
            class_result, postprocessor.classes_report()
        )
        auc_result = Postprocessing.update_report(
            auc_result, postprocessor.auc_report()
        )
        roc_result = Postprocessing.update_report(
            roc_result, postprocessor.roc_report()
        )

if not isdir(RESULT_BASEDIR):
    makedirs(RESULT_BASEDIR)
train_result.to_csv(train_result_file(RESULT_BASEDIR, GRAPHNAME))
class_result.to_csv(class_result_file(RESULT_BASEDIR, GRAPHNAME))
auc_result.to_csv(auc_result_file(RESULT_BASEDIR, GRAPHNAME))
roc_result.to_csv(roc_result_file(RESULT_BASEDIR, GRAPHNAME))
