from typing import Literal

import dspy
import modaic


class ToxicityJudge(dspy.Signature):
    """Classify the toxicity of a user message request"""

    message: str = dspy.InputField()
    category: Literal[
        "self-harm",
        "deception",
        "harassment",
        "sexual",
        "violence",
        "unethical",
        "privacy",
        "hate",
        "illegal",
        "harmful",
    ] = dspy.OutputField()


arbiter = modaic.Predict(
    ToxicityJudge, lm=dspy.LM("openrouter/openai/gpt-oss-120b")
).as_arbiter()

# Replace "<username>" with your Modaic Hub username
arbiter.push_to_hub("<username>/toxicity")
