from datasets import load_dataset
from modaic import Arbiter

dataset = load_dataset("UniqueData/email-spam-classification")["train"]

arbiter = Arbiter("tyrin/spam-classification")


def add_prediction(row):
    result = arbiter.predict(subject=row["title"], body=row["text"])
    row["predicted"] = result.output.is_spam
    return row


dataset = dataset.map(add_prediction)

dataset.save_to_disk("predictions")
