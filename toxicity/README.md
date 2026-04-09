# LLM Toxicity Judge

This example walks through building a toxicity classification arbiter using [or-bench-toxic](https://huggingface.co/datasets/bench-llm/or-bench/viewer/or-bench-toxic) — a benchmark dataset of user message requests labeled with one of the following categories:

- `self-harm`, `deception`, `harassment`, `sexual`, `violence`, `unethical`, `privacy`, `hate`, `illegal`, `harmful`

You will learn how to:
1. Define and publish an arbiter to Modaic Hub
2. Run the arbiter over a real dataset
3. Use Modaic's **Calibrate** feature to generate confidence scores and catch mislabeled predictions

## Step 1. Define the Arbiter

[arbiter.py](./arbiter.py)

We use `dspy.Signature` to define the judge. The output field is constrained to the allowed labels using `typing.Literal`, which Modaic uses to enforce structured outputs.

```python
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


# Wrap the signature in modaic.Predict, then call .as_arbiter() to register it
arbiter = modaic.Predict(
    ToxicityJudge, lm=dspy.LM("openrouter/openai/gpt-oss-120b")
).as_arbiter()

# Push to Modaic Hub — replace "tyrin" with your username
arbiter.push_to_hub("tyrin/toxicity")
```

Run the script to publish the arbiter:

```bash
uv run arbiter.py
```

This creates a new repository on [Modaic Hub](https://www.modaic.dev/). Make sure `MODAIC_TOKEN` is set in your `.env` file before running.

## Step 2. Generate Predictions

[predict.py](./predict.py)

Now we load the dataset, run the arbiter over 100 examples, and save the predictions to disk.

```python
from datasets import load_dataset
from modaic import Arbiter
from modaic_client import configure_modaic_client

configure_modaic_client(timeout=120.0)

dataset = load_dataset("bench-llm/or-bench", "or-bench-toxic", split="train")

# Randomly select 100 examples
dataset = dataset.shuffle(seed=42).select(range(100))

# Load the arbiter from Modaic Hub
arbiter = Arbiter("tyrin/toxicity")


def add_prediction(row, idx):
    result = arbiter.predict(message=row["prompt"])
    row["prediction"] = result.output.category
    row["example_id"] = idx
    return row


dataset = dataset.map(add_prediction, with_indices=True)

dataset.save_to_disk("./data/or-bench-predictions")
```

Run the script:

```bash
uv run predict.py
```

The predictions will be saved to `./data/or-bench-predictions`.

## Step 3. Calibrate Confidence Scores

Once the predictions are uploaded, open the repository on [Modaic Hub](https://www.modaic.dev/) and navigate to the **Annotations** tab. Click **Calibrate** to generate a confidence score for each prediction. This may take a few minutes.

When complete, you'll see confidence scores alongside each annotation. By default, Modaic highlights predictions below your configured confidence threshold — you can adjust this in the repository settings.

![post-confidence](./assets/post-confidence.png)

## Results

Let's look at a couple of the lowest-confidence predictions to see what the confidence estimator caught.

**Example 1 — caught mislabel**

The judge classified this example as `harassment` with a confidence of 28.1%.

![harassment-mislabel](./assets/harassment-mislabel.png)

Checking the ground truth, the correct label was `violence`.

![correct label](./assets/correct_label.png)

The confidence estimator flagged a real mistake here.

**Example 2 — correct but uncertain**

The judge classified this next example as `harassment` and got it right — but the low confidence score still surfaced it for review.

![correctly labeled](./assets/harassment-correct.png)

Looking at the reasoning, the judge was torn between `violence` and `harassment`. The prediction was correct, but the uncertainty was genuine and worth knowing about.

## Going Further: Align

This example covered defining an arbiter, running it over real data, and using **Calibrate** to surface low-confidence predictions. The next step is **Align**: after you review and label a handful of flagged examples, click **Align** to fine-tune both the judge and the confidence estimator to your labeling preferences. The more feedback you provide, the better the arbiter gets — both at making the right call and at knowing when it's uncertain.
