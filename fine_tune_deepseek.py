import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Check if MPS (Metal) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
print(f"🔥 Using device: {device}")

# Load DeepSeek model (force CPU if MPS is unstable)
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map={"": device}, 
    torch_dtype=torch.float32  # MPS does not support bfloat16
)

# Apply LoRA for low-memory optimization (without bitsandbytes)
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"],  # Ensure these exist in the model
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
model.to(device)  # Ensure the model is moved to MPS

# Load dataset
dataset = load_dataset("json", data_files="cot_dataset.json")

# Preprocessing function to tokenize the dataset
def preprocess_function(examples):
    inputs = [q for q in examples["question"]]  # Extract questions
    labels = [a for a in examples["cot_answer"]]  # Extract answers

    # Tokenize input and labels
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

    labels = tokenizer(labels, padding="max_length", truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["question", "cot_answer"])

# Training arguments optimized for Mac M1/M2
training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek",
    per_device_train_batch_size=1,  # Keep batch size low to avoid memory issues
    gradient_accumulation_steps=4,  # Helps with stability
    learning_rate=2e-4,
    num_train_epochs=3,
    optim="adamw_torch",  # Use default PyTorch optimizer (MPS compatible)
    remove_unused_columns=False,  # Allow Trainer to use custom dataset columns
    logging_dir="./logs",
    logging_steps=10,  # Log progress
    save_strategy="epoch"  # Save after each epoch
)

# Fine-tune using Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

print("🚀 Starting fine-tuning...")
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_deepseek")
tokenizer.save_pretrained("./fine_tuned_deepseek")

print("✅ Fine-tuning complete! Model saved in './fine_tuned_deepseek'")
