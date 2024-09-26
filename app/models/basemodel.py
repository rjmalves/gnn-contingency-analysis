import os
from abc import abstractmethod
from typing import Type

import torch
from torch import nn

from app.internal.modeldefinition import ModelDefinition
from app.utils.singleton import Singleton


class BaseModel(nn.Module):
    """ """

    def __init__(self, model_name: str = "", *args, **kwargs) -> None:
        self.model_name = model_name
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def describe(self, batch_size: int, file_output: bool = True):
        pass

    def save(self):
        os.makedirs(f"./outputs/{self.model_name}", exist_ok=True)
        torch.save(
            self.state_dict(), f"./outputs/{self.model_name}/model_scripted.pt"
        )

    def save_checkpoint(self, optimizer: torch.optim.Optimizer):
        os.makedirs(f"./outputs/{self.model_name}", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"./outputs/{self.model_name}/model_checkpoint.pt",
        )

    def load(self):
        self.load_state_dict(
            torch.load(
                f"./outputs/{self.model_name}/model_scripted.pt",
            )
        )
        self.eval()

    def load_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        checkpoint = torch.load(
            f"./outputs/{self.model_name}/model_checkpoint.pt",
            weights_only=True,
        )
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Forces optimizer to use the same dtype as the model
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v
        # Default checkpoint model to train mode
        self.train()

    def clear_checkpoint(self):
        path = f"./outputs/{self.model_name}/model_checkpoint.pt"
        if os.path.isfile(path):
            os.remove(path)


class ModelFactory(metaclass=Singleton):
    def __init__(self) -> None:
        self._models: dict[str, Type[BaseModel]] = {}

    def register(self, model_kind: str, model: Type[BaseModel]) -> None:
        self._models[model_kind] = model

    def factory(self, definition: ModelDefinition) -> BaseModel:
        model_kind = definition.kind
        data = definition.parameters
        if model_kind in self._models:
            m = self._models[model_kind](**data)
            print(
                "Number of parameters: ", sum(p.numel() for p in m.parameters())
            )
            return m
        else:
            raise ValueError(f"Model kind {model_kind} not found")
