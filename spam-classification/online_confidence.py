from modaic import Arbiter

arbiter = Arbiter("tytodd/spam-classification")  # Replace 'tytodd' with your username
result = arbiter(
    subject="You won a free flight to paris",
    body="Just wanted to let you know you won a free flight to paris, claim it here...",
)

print(result)
