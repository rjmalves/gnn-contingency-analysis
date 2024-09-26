import torch
import torch.nn as nn


def dtype_factory(dtype: str) -> torch.dtype:
    mappings = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    default = torch.float32
    return mappings.get(dtype, default)


def activation_factory(kind: str) -> nn.Module:
    mappings = {
        "sigmoid": torch.nn.Sigmoid(),
        "tanh": torch.nn.Tanh(),
        "elu": torch.nn.ELU(),
        "relu": torch.nn.ReLU(),
        "gelu": torch.nn.GELU(),
    }
    default = torch.nn.ReLU()
    return mappings.get(kind, default)


def optimizer_factory(
    optimizer: dict, model: nn.Module
) -> torch.optim.Optimizer:
    mappings = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    default = torch.optim.Adam
    return mappings.get(optimizer["kind"], default)(
        model.parameters(), **optimizer["parameters"]
    )


def loss_factory(kind: str) -> torch.nn.modules.loss._Loss:
    mappings = {
        "mse": torch.nn.MSELoss(),
    }
    default = torch.nn.MSELoss()
    return mappings.get(kind, default)
