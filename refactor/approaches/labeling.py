from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np


class AbstractLabeling(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def label(self, criticalities: Dict[tuple, Any]) -> Dict[tuple, int]:
        pass

    @property
    @abstractmethod
    def identifier(self) -> Tuple[str, float]:
        pass


class ThresholdLabeling(AbstractLabeling):
    def __init__(self, threshold: float) -> None:
        self.__threshold = threshold
        super().__init__()

    def label(self, criticalities: Dict[tuple, Any]) -> Dict[tuple, int]:
        classes = {k: 0 for k in criticalities.keys()}
        nonzeros = [d for d in list(criticalities.values()) if d > 0]
        max_criticality = max(nonzeros)
        min_criticality = min(nonzeros)
        for d in criticalities.keys():
            if (criticalities[d] - min_criticality) / (
                max_criticality - min_criticality
            ) >= self.__threshold:
                classes[d] = 1
            elif criticalities[d] > 0:
                classes[d] = 0
            else:
                classes[d] = -1
        return classes

    @property
    def identifier(self) -> Tuple[str, float]:
        return "threshold", self.__threshold


class QuantileLabeling(AbstractLabeling):
    def __init__(self, quantile: float) -> None:
        self.__quantile = quantile
        super().__init__()

    def label(self, criticalities: Dict[tuple, Any]) -> Dict[tuple, int]:
        classes = {k: 0 for k in criticalities.keys()}
        nonzeros = np.array([d for d in list(criticalities.values()) if d > 0])
        threshold = np.quantile(nonzeros, 1 - self.__quantile)
        for d in criticalities.keys():
            if criticalities[d] >= threshold:
                classes[d] = 1
            elif criticalities[d] > 0:
                classes[d] = 0
            else:
                classes[d] = -1
        return classes

    @property
    def identifier(self) -> Tuple[str, float]:
        return "quantile", self.__quantile
