#!/bin/bash
python3 upload_to_huggingface.py
pip3 install transformers evaluate
python3 benchmark_deepseek.py