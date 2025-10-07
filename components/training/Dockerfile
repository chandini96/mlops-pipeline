# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the multi-model training script
COPY multi_model_training.py /app/multi_model_training.py

# Install Python dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib gcsfs

# Set entrypoint to the training script
ENTRYPOINT ["python", "/app/multi_model_training.py"]
