from datasets import load_dataset
from modaic import Arbiter

dataset = load_dataset("UniqueData/email-spam-classification")["train"]

# User Arbiter to run your arbiter via modaic's backend (make sure you have set a TOGETHER_API_KEY in Settings > Environment Variables)
# # Replace tyrin with your username
arbiter = Arbiter("modaic/spam-classification-modaic2")

examples = [
    {"input": {"subject": row["title"], "body": row["text"]}} for row in dataset
]

results = arbiter.predict_all(examples)

# predict_all returns results in completion order — align by the preserved input
predicted_by_input = {
    (r.input["subject"], r.input["body"]): r.predictions[0].output.is_spam
    for r in results
}

dataset = dataset.map(
    lambda row: {"predicted": predicted_by_input[(row["title"], row["text"])]}
)

dataset.save_to_disk("predictions")
