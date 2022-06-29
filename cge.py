import networkx as nx
import pandas as pd
from itertools import product
import torch
from typing import List
from os import getenv, makedirs
from os.path import isdir, join
from dotenv import load_dotenv
from refactor.approaches.cge import (
    CGE,
    train_embedding,
    train_classification,
    test,
)


from refactor.approaches.labeling import QuantileLabeling
from refactor.approaches.postprocessing import Postprocessing
from refactor.approaches.preprocessing import Preprocessing
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
RESULT_BASEDIR = join(getenv("RESULT_BASEDIR"), "cge")
FIGURE_BASEDIR = join(RESULT_BASEDIR, "figures")
K = [int(k) for k in getenv("K").split(",") if len(k) > 0]
EDGELIST = edgelist_file(EDGELIST_BASEDIR, GRAPHNAME)
TOL = [0.25]
N_EVALS = 1

# Training parameters
TRAIN_SPLIT = [0.5]
NUM_EPOCHS = 100
LEARNING_RATE = 1e-2

# Model parameters
EMBEDDING_D = 8
WALK_LENGTH = 10
CONTEXT_WINDOW_SIZE = 10
WALKS_PER_NODE = 10
NUMBER_OF_NEGATIVE_SAMPLES = 1
P = 1
Q = 1

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
        data = preprocessor.torch_edge_data

        model = CGE(
            data=data,
            embedding_dimension=EMBEDDING_D,
            walk_length=WALK_LENGTH,
            context_window_size=CONTEXT_WINDOW_SIZE,
            walks_per_node=WALKS_PER_NODE,
            number_of_negative_samples=NUMBER_OF_NEGATIVE_SAMPLES,
            p=P,
            q=Q,
        )

        optimizer = torch.optim.SparseAdam(
            list(model.embedding_model.parameters()), lr=LEARNING_RATE
        )

        train_losses: List[float] = []
        val_losses: List[float] = []
        train_edges, test_edges = preprocessor.train_test_edges
        edges_labels = preprocessor.edges_labels
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, val_loss = train_embedding(
                model, data, optimizer, train_edges, test_edges, edges_labels
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f},  Val Loss: {val_loss:.4f}"
                )

        train_classification(model, train_edges, edges_labels)

        y, yhat = test(model, test_edges, edges_labels)
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
