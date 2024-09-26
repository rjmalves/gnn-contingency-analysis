from contextlib import nullcontext

import torch

from app.internal.taskhandler import AbstractTaskHandler
from app.tasks.training.definition import TrainingDefinition
from app.utils.factories import (
    dtype_factory,
)


class TrainingHandler(AbstractTaskHandler):
    @property
    def ctx(self):
        definition: TrainingDefinition = self._task_definition

        device_type = "cuda" if "cuda" in definition.device else "cpu"
        ctx = (
            nullcontext()
            if definition.device == "cpu"
            else torch.autocast(
                device_type=device_type,
                dtype=dtype_factory(definition.dtype),
            )
        )
        return ctx

    def preprocess(self, *args, **kwargs):
        """
        Input data processing and instantiating the necessary data
        structures for training and testing the models.

        - Instantiates the datasets and dataloaders
        - The dataset is complex enough so that it can handle the
            downloading of necessary data, enumerating time windows
            and assembling samples
        - The dataloaders iterate on the dataset, preparing data for
            the model

        """
        definition: TrainingDefinition = self._task_definition
