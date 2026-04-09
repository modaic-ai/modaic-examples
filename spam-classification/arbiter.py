import dspy
import modaic


class SpamClassifier(dspy.Signature):
    # The docstring of the dspy.Signature will be used in the system prompt of the Arbiter
    """Classify the email as spam or not spam."""

    # Define input fields
    subject: str = dspy.InputField()
    body: str = dspy.InputField()

    # Define output field
    is_spam: bool = dspy.OutputField(desc="Whether the message is spam or not spam.")


# Create a an Arbiter by initializing a modaic.Predict with the defined Signature and using .as_arbiter
classifier = modaic.Predict(
    SpamClassifier, lm=modaic.SafeLM(model="together_ai/openai/gpt-oss-120b")
).as_arbiter()

# Lets run a quick example offline to see how the predict classifies an example
result = classifier(
    subject="You won a free flight to paris",
    body="Dear Sir or Madam, \n We have been trying to contact you regarding your car's extended warranty",
    return_messages=True,
)
print("Is Spam:", result.is_spam)
print("Messages:", result._messages)
print("Outputs:", result._outputs)

# Push arbiter to modaic hub (replace tyrin with your modaic username)
classifier.push_to_hub("tyrin/spam-classification")
