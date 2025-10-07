# components/feature_engineer/Dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==1.24.3 pandas==1.5.3 scikit-learn==1.2.2 gcsfs==2023.6.0 joblib==1.3.0
COPY feature_engineer.py /app/feature_engineer.py

# Set ENTRYPOINT to run the script directly
ENTRYPOINT ["python3", "/app/feature_engineer.py"]