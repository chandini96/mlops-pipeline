# Build and Push All Docker Images for MLOps Pipeline

Write-Host "Building and pushing all Docker images..." -ForegroundColor Green

# 1. Data Fetcher
Write-Host "Building data fetcher image..." -ForegroundColor Yellow
Set-Location components/data_fetching
docker build -t us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-data-fetcher:v1.0.6 .
docker push us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-data-fetcher:v1.0.6
Set-Location ../..

# 2. Data Preprocessor
Write-Host "Building data preprocessor image..." -ForegroundColor Yellow
Set-Location components/preprocessing
docker build -t us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-data-preprocessor:v1.0.7 .
docker push us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-data-preprocessor:v1.0.7
Set-Location ../..

# 3. Feature Engineer
Write-Host "Building feature engineer image..." -ForegroundColor Yellow
Set-Location components/feature_engineering
docker build -t us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-feature-engineer:v1.0.3 .
docker push us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-feature-engineer:v1.0.3
Set-Location ../..

# 4. Data Splitter
Write-Host "Building data splitter image..." -ForegroundColor Yellow
Set-Location components/data_splitting
docker build -t us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-data-splitter:v1.0.0 .
docker push us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-data-splitter:v1.0.0
Set-Location ../..

# 5. Model Trainer
Write-Host "Building model trainer image..." -ForegroundColor Yellow
Set-Location components/training
docker build -t us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-model-trainer:v1.0.1 .
docker push us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-model-trainer:v1.0.1
Set-Location ../..

# 6. Model Registry
Write-Host "Building model registry image..." -ForegroundColor Yellow
Set-Location components/model_registry
docker build -t us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-model-registry:v1.0.0 .
docker push us-central1-docker.pkg.dev/pro-vigil-dev/mlops/mlops-model-registry:v1.0.0
Set-Location ../..

Write-Host "All Docker images built and pushed successfully!" -ForegroundColor Green
Write-Host "You can now run the pipeline again." -ForegroundColor Green 