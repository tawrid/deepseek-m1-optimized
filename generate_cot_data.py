import json
import time
import ollama  # Requires 'ollama' CLI

# Define sample questions
questions = [
    "Explain recursion in Python with step-by-step reasoning.",
    "Solve 24 × 17 using chain-of-thought reasoning.",
    "Describe how RSA encryption works step by step.",
    "How does the quicksort algorithm work?",
    "Explain the Pythagorean theorem with a detailed proof."
]

cot_data = []
print("🧠 Generating CoT dataset from DeepSeek...")

for q in questions:
    print(f"🔍 Asking: {q}")
    response = ollama.chat(model="deepseek-coder", messages=[{"role": "user", "content": q}])
    cot_data.append({"question": q, "cot_answer": response["message"]["content"]})
    time.sleep(2)  # Avoid rate limiting

# Save dataset
with open("cot_dataset.json", "w") as f:
    json.dump(cot_data, f, indent=4)

print("✅ CoT dataset saved as 'cot_dataset.json'")