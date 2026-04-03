from dotenv import load_dotenv

from scorer import correctness_scorer

load_dotenv()

input = "What is the capital of Germany?"
output = "Paris"
result = correctness_scorer(input, output)
print(result)
