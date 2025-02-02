#!/bin/bash

# Install Minikube, kubectl, and Docker (Make sure Docker is ARM-compatible)
brew install minikube kubectl docker

# Install Python dependencies
pip3 install torch torchvision torchaudio transformers accelerate datasets peft huggingface_hub

# Log in to Hugging Face CLI
huggingface-cli login

# Start Docker Desktop (Ensure Docker is running first)
open -a Docker

# Wait for Docker to initialize (you can tweak this based on your system)
while ! docker info >/dev/null 2>&1; do
  echo "Waiting for Docker to start..."
  sleep 5
done

# Start Minikube with Docker as the driver and sufficient resources
minikube start --memory=6000 --cpus=4 --driver=docker

# Create the 'deepseek' namespace in Kubernetes
kubectl create namespace deepseek
