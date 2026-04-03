# Braintrust Integration Demo
This demo shows how to create an arbiter on modaic and use it as a custom scorer on braintrust.

## Step 1: Define Modaic Arbiter
First define your arbiter. Here we will define a simple arbiter that takes some input to an LLM and the output it produced and determine if the LLM's response was satisfactory.
[arbiter.py](./arbiter.py)
```python
from typing import Literal

import dspy
import modaic


# Use DSPy signature to define the inputs, outputs and instuctions to the arbiter
class ResponseQuality(dspy.Signature):
    # The docstring is the system prompt that will be sent to the LLM
    """
    Determine if the LLM's response was satisfactory or not.
    """

    # Inputs to the arbiter use dspy.InputField
    input: str = dspy.InputField(desc="Input to LLM")
    output: str = dspy.InputField(desc="Output of LLM")

    # Outputs from the arbiter use dspy.OutputField
    satisfactory: bool = dspy.OutputField()


# Now we create our arbiter using modaic.Predict
quality_arbiter = modaic.Predict(
    ResponseQuality,
    lm=dspy.LM(
        model="together_ai/openai/gpt-oss-120b"
    ),  # Specify the LLM the model should use
).as_arbiter()  # convert to an arbiter

quality_arbiter.push_to_hub(
    "<username>/satisfactory"
)  # push the arbiter to the hub make sure to replace <username> with your actual username
```

Run the script to push your arbiter to Modaic Hub
```bash
uv run arbiter.py
```

## Step 2: Define Braintrust Scorer
[scorer.py](./scorer.py)
```python
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
```

## Step 3. Upload Scorer to Braintrust
```bash
export BRAINTRUST_API_KEY= <Your braintrust api key>
uv run braintrust push scorer.py --requirements requirements.txt
```

## Step 4. Set MODIAC_TOKEN Environemnt Variable
Grab a Modaic Token by clicking on your profile then selecting Access Tokens. You will need to add this to Braintrust. Open braintrust then navigate to Settings > Env Variables and add your `MODAIC_TOKEN` as an environemnt variable.

## Step 5. Add your LLM API Key to Modaic
Click on your profile icon then navigate to Settings > Envrionment Variables, then add the appropriate LiteLLM environment variable for your model. In this example the environment variable is called TOGETHERAI_API_KEY and should be set to your api key from (together.ai)[https://www.together.ai/]

You should now be able to run your custom scorer on braintrust.
