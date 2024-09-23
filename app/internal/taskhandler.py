from abc import ABC, abstractmethod
from typing import Type

from app.data.dataset import DatasetDefinition
from app.data.sampling import SamplingDefinition
from app.internal.modeldefinition import ModelDefinition
from app.internal.taskdefinition import AbstractTaskDefinition
from app.internal.taskresult import AbstractTaskResult
from app.utils.singleton import Singleton


class AbstractTaskHandler(ABC):
    def __init__(
        self,
        task_definition: AbstractTaskDefinition,
        dataset_definition: DatasetDefinition,
        sampling_definition: SamplingDefinition,
        model_definition: ModelDefinition,
    ) -> None:
        self._task_definition = task_definition
        self._dataset_definition = dataset_definition
        self._sampling_definition = sampling_definition
        self._model_definition = model_definition

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self) -> AbstractTaskResult:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        raise NotImplementedError


class TaskHandlerFactory(metaclass=Singleton):
    def __init__(self) -> None:
        self._tasks: dict[str, Type[AbstractTaskHandler]] = {}

    def register(self, task_kind: str, task: Type[AbstractTaskHandler]) -> None:
        self._tasks[task_kind] = task

    def factory(
        self,
        task_kind: str,
        task_definition: AbstractTaskDefinition,
        dataset_definition: DatasetDefinition,
        sampling_definiton: SamplingDefinition,
        model_definition: ModelDefinition,
    ) -> AbstractTaskHandler:
        if task_kind in self._tasks:
            return self._tasks[task_kind](
                task_definition,
                dataset_definition,
                sampling_definiton,
                model_definition,
            )
        else:
            raise ValueError(f"Task kind {task_kind} not found")
