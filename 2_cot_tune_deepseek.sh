#!/bin/bash

# Generate CoT dataset
echo "📊 Generating Chain-of-Thought dataset..."
python3 generate_cot_data.py

# Start fine-tuning
echo "🎯 Starting fine-tuning..."
python3 fine_tune_deepseek.py