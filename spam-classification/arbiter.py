import dspy
import modaic


class SpamClassifier(dspy.Signature):
    """Classify the email as spam or not spam."""

    subject: str = dspy.InputField()
    body: str = dspy.InputField()
    is_spam: bool = dspy.OutputField(desc="Whether the message is spam or not spam.")


classifier = modaic.Predict(
    SpamClassifier, lm=modaic.SafeLM(model="together_ai/openai/gpt-oss-120b")
).as_arbiter()


result = classifier(
    subject="You won a free flight to paris",
    body="Dear Sir or Madam, \n We have been trying to contact you regarding your car's extended warranty",
    return_messages=True,
)
print("Is Spam:", result.is_spam)
print("Messages:", result._messages)
print("Outputs:", result._outputs)

# classifier.push_to_hub("tytodd/spam-classification")
classifier.push_to_hub("tyrin/spam-classification")
