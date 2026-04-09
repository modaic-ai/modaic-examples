from datasets import load_dataset
from modaic import Arbiter
from modaic_client import configure_modaic_client

configure_modaic_client(timeout=120.0)

dataset = load_dataset("bench-llm/or-bench", "or-bench-toxic", split="train")

# Randomly select 100 examples
dataset = dataset.shuffle(seed=42).select(range(100))

# Replace "tyrin" with your Modaic Hub username
arbiter = Arbiter("tyrin/toxicity")


def add_prediction(row, idx):
    result = arbiter.predict(message=row["prompt"])
    row["prediction"] = result.output.category
    row["example_id"] = idx
    return row


dataset = dataset.map(add_prediction, with_indices=True)

dataset.save_to_disk("./data/or-bench-predictions")
