from dataclasses import dataclass

from app.internal.taskdefinition import (
    AbstractTaskDefinition,
    TaskDefinitionFactory,
)


@dataclass
class TrainingDefinition(AbstractTaskDefinition):
    seed: int
    train_val_test_splits: list[float]
    batch_size: int
    num_epochs: int
    loss_function: str
    optimizer: dict
    device: str
    dtype: str
    debug: dict
    output_column: str
    output: dict
    resume_training: bool

    @classmethod
    def from_json(cls, data: dict) -> "TrainingDefinition":
        return cls(
            seed=data["seed"],
            train_val_test_splits=data["train_val_test_splits"],
            batch_size=data["batch_size"],
            num_epochs=data["num_epochs"],
            loss_function=data["loss_function"],
            optimizer=data["optimizer"],
            debug=data["debug"],
            output=data["output"],
            output_column=data["output_column"],
            device=data["device"],
            dtype=data["dtype"],
            resume_training=data["resume_training"],
        )

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "train_val_test_splits": self.train_val_test_splits,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "loss_function": self.loss_function,
            "optimizer": self.optimizer,
            "device": self.device,
            "dtype": self.dtype,
            "debug": self.debug,
            "output_column": self.output_column,
            "output": self.output,
            "resume_training": self.resume_training,
        }


TaskDefinitionFactory().register("training", TrainingDefinition)
