import dspy
import modaic


class SpamClassifier(dspy.Signature):
    """Classify the email as spam or not spam."""

    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    is_spam: bool = dspy.OutputField(desc="Whether the message is spam or not spam.")


classifier = modaic.Predict(
    SpamClassifier, lm=dspy.LM(model="together_ai/openai/gpt-oss-120b")
).as_arbiter()

classifier.push_to_hub("tytodd/spam-classification")
