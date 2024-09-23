from abc import ABC, abstractmethod
from typing import Type

from app.utils.singleton import Singleton


class AbstractTaskDefinition(ABC):
    @classmethod
    @abstractmethod
    def from_json(cls, data: dict):
        raise NotImplementedError

    @abstractmethod
    def to_json(self) -> dict:
        raise NotImplementedError


class TaskDefinitionFactory(metaclass=Singleton):
    def __init__(self) -> None:
        self._tasks: dict[str, Type[AbstractTaskDefinition]] = {}

    def register(
        self, task_kind: str, task: Type[AbstractTaskDefinition]
    ) -> None:
        self._tasks[task_kind] = task

    def factory(self, task_kind: str, data: dict) -> AbstractTaskDefinition:
        if task_kind in self._tasks:
            return self._tasks[task_kind].from_json(data)
        else:
            raise ValueError(f"Task kind {task_kind} not found")
