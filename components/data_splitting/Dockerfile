# components/data_splitting/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy==1.24.3 \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    gcsfs==2023.6.0

# Copy data splitting script
COPY data_splitter.py /app/data_splitter.py

# Set ENTRYPOINT to run the script directly
ENTRYPOINT ["python3", "/app/data_splitter.py"]