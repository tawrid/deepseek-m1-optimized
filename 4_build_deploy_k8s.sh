#!/bin/bash

# Ensure the environment variables for Minikube's Docker daemon are set
eval $(minikube -p minikube docker-env)

# Build the Docker image for ARM64 (ensure it's compatible with M1/M2)
docker login

docker build -t tawrid/deepseek-finetuned:latest .

docker push tawrid/deepseek-finetuned:latest

# Load the image into Minikube's Docker daemon
minikube image load deepseek-finetuned

kubectl create secret generic deepseek-finetuned-registry-key \
  --from-file=.dockerconfigjson=$HOME/.docker/config.json \
  --type=kubernetes.io/dockerconfigjson

# Apply the Kubernetes deployment and service configurations
kubectl apply -f deepseek_deployment.yaml
kubectl apply -f deepseek_service.yaml

# Get services and ensure the deployment is successful
kubectl get services -n deepseek
