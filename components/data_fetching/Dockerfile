# components/data_fetcher/Dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==1.24.3 pandas==1.5.3 gcsfs==2023.6.0
COPY data_fetcher.py /app/data_fetcher.py

# Set ENTRYPOINT to run the script directly
ENTRYPOINT ["python3", "/app/data_fetcher.py"] 