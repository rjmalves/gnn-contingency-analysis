from app.data.sources import FSDataSource
from app.data.dataset import DatasetDefinition
from app.data.sampling import SamplingDefinition
from app.utils.singleton import Singleton
from app.utils.factories import activation_factory
from app.internal.modeldefinition import ModelDefinition
from app.internal.taskdefinition import (
    AbstractTaskDefinition,
    TaskDefinitionFactory,
)
from datetime import datetime
from ray import tune
import json


class Settings(metaclass=Singleton):
    """
    Settings that are used for the studies.
    """

    # Name
    name: str = "conv1d"

    # Output directory
    outdir: str = "./outputs"

    # Dataset definition
    dataset = DatasetDefinition(
        start_date=datetime(2019, 1, 20),
        end_date=datetime(2021, 12, 31),
        source=FSDataSource(path="./data", format="parquet"),
    )

    # Sample definition
    sampling = SamplingDefinition(
        lookback_window_length=7,
        horizon=1,
        stride=1,
        input_columns=[f"previsto{str(i).zfill(2)}" for i in range(9)],
        target_column="verificado",
        date_column="data_hora",
        key_column=None,
    )

    # Model definition
    _model = ModelDefinition(
        kind="Conv1dCNN",
        parameters={
            "num_in_channels": 6,
            "conv_channels": [6],
            "kernel_sizes": [25],
            "series_size": 144,
            "out_len": 48,
            "activation": "elu",
        },
    )

    tasks: list[AbstractTaskDefinition] = []

    @classmethod
    def read(cls, file: str = "settings.json") -> "Settings":
        with open(file, "r") as fp:
            data = json.load(fp)
        # Name
        cls.name = data["name"]
        # Output directory
        cls.outdir = data["outdir"]
        # Dataset definition
        cls.dataset = DatasetDefinition.factory(data["dataset"])
        # Sample definition
        cls.sampling = SamplingDefinition.factory(data["sampling"])
        # Model definition
        cls._model = ModelDefinition.factory(data["model"])
        # Tasks
        cls.tasks = data["tasks"]
        return cls()

    @classmethod
    def __get_seq_len(cls) -> int:
        return cls.sampling.lookback_window_length

    @classmethod
    def __get_out_len(cls) -> int:
        return (
            1
            if type(cls.sampling.horizon) is int
            else len(cls.sampling.horizon)
        )

    @classmethod
    def __get_model_parameters(cls) -> dict:
        params = cls._model.parameters
        additional_params = {
            "model_name": cls.name,
            "seq_len": cls.__get_seq_len(),
            "out_len": cls.__get_out_len(),
            "temporal_embedding": cls.sampling.temporal_embedding,
        }
        if "activation" in cls._model.parameters.keys():
            additional_params["activation"] = activation_factory(
                cls._model.parameters["activation"]
            )
        return {**params, **additional_params}

    @classmethod
    def get_tasks_definitions(cls) -> list[AbstractTaskDefinition]:
        return [
            TaskDefinitionFactory().factory(t["kind"], t["parameters"])
            for t in cls.tasks
        ]

    @classmethod
    def get_model_definition(cls) -> ModelDefinition:
        return ModelDefinition(
            kind=cls._model.kind, parameters=cls.__get_model_parameters()
        )

    @classmethod
    def get_tuning_model_parameters(cls) -> dict:
        params = {
            k: cls._get_parameter_config(v)
            for k, v in cls.tuning.model_parameters.items()
        }
        additional_params = {
            "model_name": cls.name,
            "seq_len": cls.__get_seq_len(),
            "out_len": cls.__get_out_len(),
        }
        if "activation" in cls.tuning.model_parameters.keys():
            additional_params["activation"] = activation_factory(
                cls.tuning.model_parameters["activation"]
            )
        return {**params, **additional_params}

    @classmethod
    def _get_parameter_config(cls, param) -> dict:
        def _choice_handler(x):
            return tune.choice(x["values"])

        mappings = {
            "choice": _choice_handler,
        }
        if isinstance(param, dict):
            return mappings[param["kind"]](param)
        else:
            return param
