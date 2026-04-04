from modaic import Arbiter

arbiter = Arbiter("tyrin/spam-classification")
# arbiter = Arbiter("tytodd/spam-classification")  # Replace 'tytodd' with your username
result = arbiter(
    subject="You won a free flight to paris",
    body="Dear Sir or Madam, \n We have been trying to contact you regarding your car's extended warranty",
)

print("Is Spam:", result.output.is_spam)
print("messages:", result.messages)
print("Confidence:", result.confidence)
