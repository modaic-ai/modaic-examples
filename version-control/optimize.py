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
