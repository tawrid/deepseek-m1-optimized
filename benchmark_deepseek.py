import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set device to MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.backends.mps.allow_tf32 = True  # Enable TensorFloat32 for better performance

# Load model and tokenizer
model_name = "./fine_tuned_deepseek"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# Apply PyTorch optimization (remove if unstable)
model = torch.compile(model)

# Define benchmark function
def benchmark():
    test_prompts = [
        "Solve 24 × 17 using step-by-step reasoning.",
        "Explain recursion in Python.",
        "Describe how RSA encryption works step by step.",
        "How many stars are in the whole universe?"
    ]

    times = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model.generate(**inputs, max_length=50, do_sample=False, num_beams=1)

        end = time.time()  # No need for extra synchronization
        elapsed_time = end - start
        times.append(elapsed_time)

        print(f"Prompt: {prompt[:30]}... | Time: {elapsed_time:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"✅ Average response time: {avg_time:.2f} seconds")

# Run benchmark
benchmark()
