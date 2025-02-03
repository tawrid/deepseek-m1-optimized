#!/bin/bash

echo "🚀 Setting up DeepSeek AI on Mac M1..."

# Update Homebrew and install dependencies
echo "🔄 Updating Homebrew..."
brew update && brew upgrade

echo "🛠 Installing dependencies..."
brew install git python3 wget
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio transformers accelerate datasets bitsandbytes peft

# Check if Metal backend is available (MPS support for PyTorch on Mac M1)
echo "✅ Checking Metal support for PyTorch..."
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Download Ollama zip and install
echo "⬇️ Downloading Ollama..."
curl -fsSL https://ollama.com/download/Ollama-darwin.zip -o Ollama-darwin.zip

echo "📦 Unzipping Ollama..."
unzip -q Ollama-darwin.zip -d ollama

echo "📂 Moving Ollama to Applications..."
mv ollama/Ollama.app /Applications/

Pull DeepSeek model with Ollama
echo "📥 Downloading DeepSeek Coder model..."
/Applications/Ollama.app/Contents/MacOS/Ollama pull deepseek-coder:6.7b

echo "🚀 Running DeepSeek model..."
/Applications/Ollama.app/Contents/MacOS/Ollama run deepseek-coder:6.7b &  # Run in background

echo "✅ Setup complete! Run '/Applications/Ollama.app/Contents/MacOS/Ollama run deepseek-coder:6.7b' anytime to use the model."

