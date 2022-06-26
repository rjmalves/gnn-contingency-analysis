import networkx as nx
import pandas as pd
from itertools import product
import torch
from os import getenv, makedirs
from os.path import isdir
from dotenv import load_dotenv


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
)

load_dotenv(override=True)

# Study case parameters
GRAPHNAME = getenv("GRAPHNAME")
EDGELIST_BASEDIR = getenv("EDGELIST_BASEDIR")
CRITICALITY_BASEDIR = getenv("CRITICALITY_BASEDIR")
RESULT_BASEDIR = getenv("RESULT_BASEDIR")
FIGURE_BASEDIR = getenv("FIGURE_BASEDIR")
K = [int(k) for k in getenv("K").split(",") if len(k) > 0]
EDGELIST = edgelist_file(EDGELIST_BASEDIR, GRAPHNAME)
TOL = [0.05, 0.10, 0.15, 0.20, 0.25]
N_EVALS = 50

# Training parameters
TRAIN_SPLIT = [0.1]
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3

# Model parameters
EMBEDDING_D = 128
DROPOUT = 0.1
HIDDEN_CHANNELS = 64


G = nx.read_edgelist(EDGELIST)
combinations = list(product(K, TOL, TRAIN_SPLIT))

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
        data = preprocessor.torch_data

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
        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train(model, data, optimizer, criterion)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

        y, yhat = test(model, data)
        postprocessor = Postprocessing(
            y, yhat, k, train_split, labeling_strategy, eval=i
        )
        class_r = postprocessor.classes_report()
        if class_result.empty:
            class_result = class_r
        else:
            class_result = pd.concat(
                [class_result, class_r], ignore_index=True
            )
        auc_r = postprocessor.auc_report()
        if auc_result.empty:
            auc_result = auc_r
        else:
            auc_result = pd.concat([auc_result, auc_r], ignore_index=True)
        roc_r = postprocessor.roc_report()
        if roc_result.empty:
            roc_result = roc_r
        else:
            roc_result = pd.concat([roc_result, roc_r], ignore_index=True)

if not isdir(RESULT_BASEDIR):
    makedirs(RESULT_BASEDIR)
class_result.to_csv(class_result_file(RESULT_BASEDIR, GRAPHNAME))
auc_result.to_csv(auc_result_file(RESULT_BASEDIR, GRAPHNAME))
roc_result.to_csv(roc_result_file(RESULT_BASEDIR, GRAPHNAME))
