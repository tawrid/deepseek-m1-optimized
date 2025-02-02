# Use an official Python image as the base
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the model directory and the server.py file into the container
COPY fine_tuned_deepseek /app/fine_tuned_deepseek
COPY server.py /app/server.py

# Install the necessary Python dependencies
RUN pip install --no-cache-dir fastapi transformers torch uvicorn

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Set environment variables for GPU (optional)
ENV CUDA_VISIBLE_DEVICES="0" 
# Set based on available GPUs or remove if using CPU

# Set the default command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
