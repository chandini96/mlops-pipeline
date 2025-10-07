# components/model_registry/Dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas gcsfs joblib google-cloud-aiplatform
COPY model_registry.py /app/model_registry.py

# Set ENTRYPOINT to run the script directly
ENTRYPOINT ["python3", "/app/model_registry.py"] 