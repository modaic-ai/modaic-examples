import dspy

from program import DocExtract

loaded_program = DocExtract.from_precompiled("<username>/doc-extract")
print("Loaded program prompt:", loaded_program.cot.predict.signature.instructions)

dspy.configure(lm=dspy.LM(model="openai/gpt-5-mini"))
result = loaded_program(context="This is a legal document")

print("Label:", result.label)
