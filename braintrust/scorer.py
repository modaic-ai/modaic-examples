import os

import braintrust
from modaic_client import Arbiter
from pydantic import BaseModel

project = braintrust.projects.create(
    name="My Project 2"
)  # name can be new project or existing project


class CorrectnessParams(BaseModel):
    input: dict
    output: dict


def correctness_scorer(input: dict, output: dict):
    arbiter = Arbiter(
        "<username>/satisfactory"
    )  # replace <username> with your actual username
    result = arbiter.predict(input=input, output=output)
    return result.output.satisfactory


project.scorers.create(
    name="Satisfactory Scorer",
    slug="satisfactory-scorer",
    description="Check if the output is satisfactory",
    parameters=CorrectnessParams,
    handler=correctness_scorer,
    metadata={"__pass_threshold": 0.5},
)
