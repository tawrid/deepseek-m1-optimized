# deepseek-m1-optimized

## Fine-tuned DeepSeek Coder on Mac M1 & Kubernetes Deployment



This repository contains a fine-tuned version of the DeepSeek Coder model, optimized for Mac M1 and deployed on a Kubernetes cluster. The model is fine-tuned using QLoRA and trained with a Chain-of-Thought dataset for improved reasoning capabilities.



### 🚀 Features

• Fine-tuned DeepSeek Coder 6.7B model

• Optimized for Apple M1/M2 (Metal backend)

• Supports QLoRA for low-memory adaptation

• Deployment via Docker & Kubernetes

• Benchmarking script for performance evaluation

### 📥 Installation & Setup



1. Clone the Repository


```
git clone https://github.com/tawrid/deepseek-finetuned.git

cd deepseek-finetuned
```
2. Install Dependencies



Ensure you have Homebrew installed, then run:

```
brew update && brew upgrade
brew install git python3 wget
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio 
```
transformers accelerate datasets bitsandbytes peft

3. Install Ollama (Mac M1)



Since ollama is only available via a ZIP file, install it manually:

```
curl -o ollama.zip https://ollama.com/download/Ollama-darwin.zip
unzip ollama.zip -d /Applications
```
### 🔥 Fine-Tuning the Model



Run the fine-tuning script using:

```
python3 fine_tune_deepseek.py
```
#### Common Errors & Fixes

• bitsandbytes was compiled without GPU support

→ Solution: Use pip install bitsandbytes --no-cache-dir

• Dataset column mismatch (No columns match model's forward method)

→ Solution: Add remove_unused_columns=False in TrainingArguments.

### 🏎 Benchmarking Performance



Test the model’s response time using:

```
python3 benchmark_deepseek.py
```
Example Output

Prompt: Solve 24 × 17 using step-by-step...
Time: 2.13 seconds
Prompt: Explain recursion in Python...
Time: 1.95 seconds

✅ Average response time: 2.04 seconds


### 📦 Docker & Kubernetes Deployment



1. Build & Load Docker Image

````
docker build -t deepseek-finetuned .
````
For Minikube, load the image:
```
minikube image load deepseek-finetuned
````
2. Deploy on Kubernetes
```
kubectl apply -f deepseek_deployment.yaml
kubectl apply -f deepseek_service.yaml
kubectl get services -n deepseek
```

If you encounter validation errors, disable validation:

```
kubectl apply -f deepseek_deployment.yaml --validate=false
```

3. Access the Model via API



Once deployed, check the running service:

```
kubectl get services -n deepseek
```
Then, send a request:

```
curl -X POST http://<SERVICE_IP>:<PORT>/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain quantum computing"}'
```
### 📜 Repository Structure

```
📂 deepseek-finetuned
├── fine_tune_deepseek.py   # Fine-tuning script
├── benchmark_deepseek.py            # Benchmarking script
├── Dockerfile              # Docker configuration
├── deepseek_deployment.yaml # Kubernetes deployment
├── deepseek_service.yaml   # Kubernetes service
└── README.md               # This file
```

### 🛠 Troubleshooting



1. How to check if Docker image is created?

```
docker images | grep deepseek-finetuned
```

2. Mac storage is full (df -h shows 100% disk usage)

• Remove unused Docker containers:

```
docker system prune -a
```


• Clear unnecessary files:

```rm -rf ~/Library/Caches/*
```
### 🤖 Contributing



Feel free to fork this repository, create a pull request, or open an issue if you find any bugs or improvements.

### 📄 License



This project is licensed under the MIT License.

This README provides clear steps for installation, fine-tuning, benchmarking, and Kubernetes deployment while also covering common errors and fixes. Let me know if you need any modifications! 🚀