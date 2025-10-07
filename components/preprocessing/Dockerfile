# components/data_preprocessor/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY data_preprocessor.py /app/data_preprocessor.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Verify installation
RUN python -c "import pandas, numpy, sklearn; print('All packages imported successfully')"

# Set ENTRYPOINT to run the script directly
ENTRYPOINT ["python3", "/app/data_preprocessor.py"]
