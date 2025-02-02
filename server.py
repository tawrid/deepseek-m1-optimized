from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "./fine_tuned_deepseek"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available

print("🔥 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.to(device)

@app.get("/")
def home():
    return {"message": "DeepSeek Fine-Tuned Model API"}

@app.post("/predict/")
def predict(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}
