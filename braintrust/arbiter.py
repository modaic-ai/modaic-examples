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
