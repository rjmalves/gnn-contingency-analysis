import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score


class Postprocessing:
    def __init__(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        k: int,
        train_split: float,
        eval: int = 0,
    ):
        self.__y = y
        self.__yhat = yhat
        self.__k = k
        self.__train_split = train_split
        self.__eval = eval

    @property
    def yhat_classes(self) -> np.ndarray:
        return self.__yhat.argmax(dim=1).detach().numpy()

    @property
    def yhat_probabilities(self) -> np.ndarray:
        return F.softmax(self.__yhat[:, :2], dim=1).detach().numpy()

    @property
    def y(self) -> np.ndarray:
        return self.__y.numpy()

    @property
    def auc(self) -> float:
        return roc_auc_score(self.y, self.yhat_probabilities[:, 1])

    @property
    def roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return roc_curve(self.y, self.yhat_probabilities[:, 1])

    def classes_report(self, threshold=None, quantile=None) -> pd.DataFrame:
        report = classification_report(
            self.y,
            self.yhat_classes,
            target_names=["regular", "critical"],
            output_dict=True,
        )
        result_dict: Dict[str, list] = {
            "eval": [self.__eval],
            "k": [self.__k],
            "train_split": [self.__train_split],
        }
        if threshold is not None:
            result_dict["threshold"] = [threshold]
        if quantile is not None:
            result_dict["quantile"] = [quantile]
        classes = [
            "regular",
            "critical",
            "macro avg",
            "weighted avg",
        ]
        measures = ["precision", "recall", "f1-score", "support"]
        for c in classes:
            for m in measures:
                col = f"{c}_{m}"
                result_dict[col] = [report[c][m]]
        result_dict["accuracy"] = [report["accuracy"]]
        return pd.DataFrame(data=result_dict)

    def auc_report(self, threshold=None, quantile=None) -> pd.DataFrame:
        result_dict: Dict[str, list] = {
            "eval": [self.__eval],
            "k": [self.__k],
            "train_split": [self.__train_split],
        }
        if threshold is not None:
            result_dict["threshold"] = [threshold]
        if quantile is not None:
            result_dict["quantile"] = [quantile]
        result_dict["auc"] = [self.auc]
        return pd.DataFrame(data=result_dict)

    def roc_report(self, threshold=None, quantile=None) -> pd.DataFrame:
        fpr, tpr, threshold = self.roc_curve
        df = pd.DataFrame(
            data={"fpr": fpr, "tpr": tpr, "threshold": threshold}
        )
        df["eval"] = self.__eval
        df["k"] = self.__k
        df["train_split"] = self.__train_split
        if threshold is not None:
            df["threshold"] = threshold
        if quantile is not None:
            df["quantile"] = quantile
        return df
