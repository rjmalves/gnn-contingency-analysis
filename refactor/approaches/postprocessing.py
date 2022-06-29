import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from refactor.approaches.labeling import AbstractLabeling
from refactor.visualization.classifications_metrics import CLASSES, METRICS


class Postprocessing:
    def __init__(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        train_losses: np.ndarray,
        val_losses: np.ndarray,
        k: int,
        train_split: float,
        labeling_strategy: AbstractLabeling,
        eval: int = 0,
    ):
        self.__y = y
        self.__yhat = yhat
        self.__train_losses = train_losses
        self.__val_losses = val_losses
        self.__k = k
        self.__train_split = train_split
        self.__labeling_strategy = labeling_strategy
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

    def train_report(self) -> pd.DataFrame:
        train_r = pd.DataFrame(
            {
                "epoch": np.arange(1, len(self.__train_losses) + 1),
                "train_loss": self.__train_losses,
                "val_loss": self.__val_losses,
            }
        )
        train_r["k"] = self.__k
        train_r["train_split"] = self.__train_split
        train_r["eval"] = self.__eval
        return train_r

    def classes_report(self) -> pd.DataFrame:
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
        name, value = self.__labeling_strategy.identifier
        result_dict[name] = value
        for c in CLASSES:
            for m in METRICS:
                col = f"{c}_{m}"
                result_dict[col] = [report[c][m]]
        result_dict["accuracy"] = [report["accuracy"]]
        return pd.DataFrame(data=result_dict)

    def auc_report(self) -> pd.DataFrame:
        result_dict: Dict[str, list] = {
            "eval": [self.__eval],
            "k": [self.__k],
            "train_split": [self.__train_split],
        }
        name, value = self.__labeling_strategy.identifier
        result_dict[name] = value
        result_dict["auc"] = [self.auc]
        return pd.DataFrame(data=result_dict)

    def roc_report(self) -> pd.DataFrame:
        fpr, tpr, threshold = self.roc_curve
        df = pd.DataFrame(
            data={"fpr": fpr, "tpr": tpr, "threshold": threshold}
        )
        df["eval"] = self.__eval
        df["k"] = self.__k
        df["train_split"] = self.__train_split
        name, value = self.__labeling_strategy.identifier
        df[name] = value
        return df

    @staticmethod
    def update_report(
        current: pd.DataFrame, new: pd.DataFrame
    ) -> pd.DataFrame:
        if current.empty:
            return new
        else:
            return pd.concat([current, new], ignore_index=True)
