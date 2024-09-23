from dataclasses import dataclass


@dataclass
class ModelDefinition:
    """
    Contains the required information for defining the model
    that will be used for the tasks.
    """

    kind: str
    parameters: dict

    @classmethod
    def factory(cls, data: dict) -> "ModelDefinition":
        return cls(
            kind=data["kind"],
            parameters=data["parameters"],
        )
