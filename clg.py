import networkx as nx
import pandas as pd
from itertools import product
import torch


from refactor.approaches.labeling import QuantileLabeling
from refactor.approaches.postprocessing import Postprocessing
from refactor.approaches.preprocessing import Preprocessing
from refactor.approaches.clg import CLG, train, test


GRAPH = "ieee39"
EDGELIST = f"/home/rogerio/git/k-contingency-screening/{GRAPH}.txt"
K = [1, 2, 3, 4]
N_EVALS = 5
TOL = [0.05, 0.10, 0.25]
TRAIN_SPLIT = [0.1]
EMBEDDING_D = 128
DROPOUT = 0.1
HIDDEN_CHANNELS = 64
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3


G = nx.read_edgelist(EDGELIST)
combinations = list(product(K, TOL, TRAIN_SPLIT))

class_result = pd.DataFrame()
auc_result = pd.DataFrame()
roc_result = pd.DataFrame()
for c in combinations:
    k, tol, train_split = c
    DELTAS = f"/home/rogerio/git/k-contingency-screening/exaustivo_{GRAPH}_{k}/edge_global_deltas.csv"
    preprocessor = Preprocessing(
        G,
        DELTAS,
        train_split=train_split,
        embedding_dimension=EMBEDDING_D,
        labeling_strategy=QuantileLabeling(tol),
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
        postprocessor = Postprocessing(y, yhat, k, train_split, eval=i)
        class_r = postprocessor.classes_report(quantile=tol)
        if class_result.empty:
            class_result = class_r
        else:
            class_result = pd.concat(
                [class_result, class_r], ignore_index=True
            )
        auc_r = postprocessor.auc_report(quantile=tol)
        if auc_result.empty:
            auc_result = auc_r
        else:
            auc_result = pd.concat([auc_result, auc_r], ignore_index=True)
        roc_r = postprocessor.roc_report(quantile=tol)
        if roc_result.empty:
            roc_result = roc_r
        else:
            roc_result = pd.concat([roc_result, roc_r], ignore_index=True)


class_result.to_csv(f"./results/clg/{GRAPH}_class.csv")
auc_result.to_csv(f"./results/clg/{GRAPH}_auc.csv")
roc_result.to_csv(f"./results/clg/{GRAPH}_roc.csv")
