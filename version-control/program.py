from typing import Literal

import dspy
from modaic import PrecompiledProgram

DocumentLabel = Literal["legal", "healthcare", "personal", "tax", "other"]


class ExtractSignature(dspy.Signature):
    """Classify a document snippet into one of the supported label buckets."""

    context: str = dspy.InputField(desc="Document text or snippet to classify")
    label: DocumentLabel = dspy.OutputField(
        desc="Return exactly one of: legal, healthcare, personal, tax, or other"
    )


# Here you can define your dspy.Module as a modaic.PrecompiledProgram
class DocExtract(PrecompiledProgram):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(ExtractSignature)

    def forward(self, context: str) -> dspy.Prediction:
        return self.cot(context=context)


if __name__ == "__main__":
    program = DocExtract()
    dspy.configure(lm=dspy.LM(model="gpt-5-mini"))
    result = program(context="This is a legal document")
    print(result.label)
    program.push_to_hub(
        "<username>/doc-extract"
    )  # push the program to modaic hub, replace with your actual username
