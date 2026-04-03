# Version Control With Modaic Hub Example

You can use the Modaic SDK + Modaic Hub to version control DSPy programs. This makes prompt management during optimization cycles very convenient.

> Before running this example refer to [example.env](./example.env) to set your `MODAIC_TOKEN` and `OPENAI_API_KEY`
## Step 1. Define your Program
You can define a program as you would any normal DSPy program. Instead of subclassing `dspy.Module`, subclass `modaic.PrecompiledProgram`. Below is an example `ChainOfThought` program that classifies documents into 5 different label categories:
- legal
- healthcare
- personal
- tax
- other

[program.py](./program.py)
```python
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
    program.push_to_hub("<username>/doc-extract") # push the program to modaic hub, replace with your actual username
```

## Step 2. Optimize your Program

Now that your program is on Modaic Hub, you can easily pull it, run an optimization job, and then push it back to the hub. Below you can see how we optimize the above program with GEPA.

[optimize.py](./optimize.py)
```python
import json

import dspy

from program import DocExtract


def load_dataset(path):
    with open(path) as f:
        return [
            dspy.Example(**json.loads(line)).with_inputs("context")
            for line in f
            if line.strip()
        ]


def classification_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> dspy.Prediction:
    del trace, pred_name, pred_trace

    if getattr(pred, "label", None) == gold.label:
        return dspy.Prediction(score=1.0, feedback="Correct.")

    return dspy.Prediction(
        score=0.0,
        feedback=f"Expected `{gold.label}` because {gold.reason}",
    )


if __name__ == "__main__":
    program = DocExtract.from_precompiled(
        "<username>/doc-extract"
    )  # replace with your actual username
    dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini", temperature=0.0))

    optimized_program = dspy.GEPA(
        metric=classification_metric,
        reflection_lm=dspy.LM("openai/gpt-4.1-mini", temperature=1.0, max_tokens=8000),
        auto="light",
    ).compile(
        program,
        trainset=load_dataset("data/train.jsonl"),
        valset=load_dataset("data/val.jsonl"),
    )

    optimized_program.push_to_hub(
        "<username>/doc-extract"
    )  # replace with your actual username
```

## Step 3. Use your Optimized Program in Production
You can pull the optimized program anywhere — offline or in production — using `<ClassName>.from_precompiled`. Below you can see the loaded program and its optimized prompt.

[use.py](./use.py)
```python
import dspy

from program import DocExtract

loaded_program = DocExtract.from_precompiled("<username>/doc-extract") # replace with your actual username
print("Loaded program prompt:", loaded_program.cot.predict.signature.instructions)

dspy.configure(lm=dspy.LM(model="openai/gpt-5-mini"))
result = loaded_program(context="This is a legal document")

print("Label:", result.label)
```
**Output**
```
Loaded program prompt: You are given a short document snippet (referred to as "context") which you must classify into one of the supported label buckets: healthcare, tax, legal, personal, or other.

Detailed guidelines for classification:

1. **Focus on Primary Topic and Purpose**
   Analyze the main subject and the principal intent of the document snippet rather than isolated keywords. Some keywords might be ambiguous or appear in multiple domains (e.g., “healthcare” terms can appear in tax-related documents).

2. **Healthcare**
   Includes documents primarily related to medical information, treatment guidance, patient care, health instructions, diagnoses, medications, health-related reports, and direct clinical or care-oriented content.

3. **Tax**
   Includes documents primarily concerned with taxation matters, such as tax filings, reimbursements with fiscal implications, deductions, tax reporting, financial records related to taxation, and any healthcare-related content that is framed in the context of taxation (e.g., healthcare reimbursements that are relevant for tax purposes).

4. **Legal**
   Pertains to legal agreements, contracts, laws, legal instructions, compliance documentation, or legal advice.

5. **Personal**
   Includes private, individual matters not related to healthcare, tax, or legal domains—such as personal correspondence, diaries, or private notes.

6. **Other**
   Covers business, marketing, product documentation, and any content that does not fall into the above categories.

7. **Disambiguation Strategy**
   - When document snippets include topics crossing domains, classify according to the dominant or primary classification relevant to the document’s context and purpose.
   - For example, although a document may mention "healthcare", if the snippet describes healthcare in the context of monetary reimbursements for tax filing purposes, classify as tax rather than healthcare.

8. **Output Requirements**
   Provide:
   - A **reasoning** paragraph explaining your classification decision, demonstrating how you determined the primary domain.
   - A **label** field stating the chosen classification (healthcare, tax, legal, personal, or other).

By following these detailed criteria and reasoning carefully about context and domain relevance, you will accurately classify document snippets into the correct label bucket.
Label: legal
```
