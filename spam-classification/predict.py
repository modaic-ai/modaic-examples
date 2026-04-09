from datasets import load_dataset
from modaic import Arbiter

dataset = load_dataset("UniqueData/email-spam-classification")["train"]

# User Arbiter to run your arbiter via modaic's backend (make sure you have set a TOGETHER_API_KEY in Settings > Environment Variables)
# # Replace tyrin with your username
arbiter = Arbiter("tyrin/spam-classification")


def add_prediction(row):
    result = arbiter.predict(subject=row["title"], body=row["text"])
    row["predicted"] = result.output.is_spam
    return row


# Map over the hf dataset getting a prediction for each row
dataset = dataset.map(add_prediction)

dataset.save_to_disk("predictions")
