from modaic import Arbiter

# Replace tyrin with your Modaic username
arbiter = Arbiter("tyrin/spam-classification")

result = arbiter(
    subject="You won a free flight to paris",
    body="Dear Sir or Madam, \n We have been trying to contact you regarding your car's extended warranty",
)

print("Is Spam:", result.output.is_spam)
print("messages:", result.messages)

# This part may take a while as it lazily sends a request to compute confidence
print("Confidence:", result.confidence)
