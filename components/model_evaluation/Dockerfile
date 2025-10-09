# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the evaluation script
COPY evaluation.py /app/evaluation.py

# Install Python dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib gcsfs

# Set the entrypoint to the evaluation script
ENTRYPOINT ["python", "/app/evaluation.py"]
