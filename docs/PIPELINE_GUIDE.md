# Kubeflow ML Pipeline Guide

This guide provides comprehensive documentation for the Kubeflow ML Pipeline project, covering everything from setup to deployment.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Components](#components)
7. [Deployment](#deployment)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)

## Overview

The Kubeflow ML Pipeline is a complete machine learning workflow that covers the entire ML lifecycle from data fetching to model training and evaluation. The pipeline is designed to be modular, scalable, and production-ready.

### Key Features

- **Modular Design**: Each component is independent and reusable
- **Data Versioning**: Tracks data versions and lineage
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Model Registry**: Centralized model management
- **Monitoring**: Performance monitoring and alerting
- **Reproducibility**: Ensures reproducible ML experiments

## Architecture

The pipeline consists of the following components:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Data Fetching │───▶│ Data Preprocessing│───▶│ Feature Engineering│
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Model Evaluation│◀───│  Model Training  │◀───│                  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
         │
         ▼
┌─────────────────┐
│ Model Deployment│
└─────────────────┘
```

### Data Flow

1. **Data Fetching**: Downloads data from external sources or APIs
2. **Data Preprocessing**: Cleans and prepares the data for training
3. **Feature Engineering**: Creates and transforms features
4. **Model Training**: Trains ML models with hyperparameter tuning
5. **Model Evaluation**: Evaluates model performance
6. **Model Deployment**: Deploys the best model (conditional)

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker
- Kubernetes cluster
- Kubeflow installation

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mlops-v2
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create directories**:
   ```bash
   mkdir -p data/{raw,processed,engineered} models evaluation logs notebooks
   ```

### Kubeflow Installation

If you haven't installed Kubeflow yet, follow these steps:

1. **Install kubectl**:
   ```bash
   # For macOS
   brew install kubectl
   
   # For Ubuntu
   sudo apt-get install kubectl
   ```

2. **Install Kubeflow**:
   ```bash
   # Using Kustomize
   kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources"
   kubectl wait --for=condition=available --timeout=600s deployment/ml-pipeline -n kubeflow
   ```

## Configuration

The pipeline is configured through `config/pipeline_config.yaml`. Key configuration sections:

### Data Configuration

```yaml
data:
  source:
    type: "api"  # api, file, database
    url: "https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv"
    local_path: "data/raw/iris.csv"
  
  preprocessing:
    test_size: 0.2
    random_state: 42
    target_column: "species"
```

### Model Configuration

```yaml
model:
  type: "classification"  # classification, regression
  algorithms:
    - "random_forest"
    - "logistic_regression"
    - "svm"
    - "xgboost"
  
  hyperparameter_tuning:
    enabled: true
    method: "grid_search"  # grid_search, random_search, bayesian
    cv_folds: 5
```

### Evaluation Configuration

```yaml
evaluation:
  metrics:
    classification:
      - "accuracy"
      - "precision"
      - "recall"
      - "f1_score"
      - "roc_auc"
  
  threshold: 0.8  # Minimum performance threshold
```

## Usage

### Running the Pipeline

1. **Create pipeline YAML**:
   ```bash
   python pipeline/main_pipeline.py
   ```

2. **Upload to Kubeflow**:
   - Open Kubeflow UI
   - Go to Pipelines
   - Upload the generated YAML file

3. **Run the pipeline**:
   - Click "Create Run"
   - Configure parameters if needed
   - Start the run

### Local Testing

```bash
# Run individual components
python components/data_fetching/data_fetcher.py
python components/preprocessing/data_preprocessor.py
python components/feature_engineering/feature_engineer.py
python components/training/model_trainer.py
python components/evaluation/model_evaluator.py
```

## Components

### 1. Data Fetching (`components/data_fetching/`)

**Purpose**: Downloads data from various sources

**Features**:
- API data fetching
- File-based data loading
- Data validation
- Error handling

**Usage**:
```python
from components.data_fetching.data_fetcher import main
main()
```

### 2. Data Preprocessing (`components/preprocessing/`)

**Purpose**: Cleans and prepares data for training

**Features**:
- Missing value handling
- Categorical encoding
- Data splitting
- Duplicate removal

**Usage**:
```python
from components.preprocessing.data_preprocessor import main
main()
```

### 3. Feature Engineering (`components/feature_engineering/`)

**Purpose**: Creates and transforms features

**Features**:
- Feature scaling
- Feature selection
- Dimensionality reduction
- Polynomial features

**Usage**:
```python
from components.feature_engineering.feature_engineer import main
main()
```

### 4. Model Training (`components/training/`)

**Purpose**: Trains multiple models with hyperparameter tuning

**Features**:
- Multiple algorithms
- Hyperparameter optimization
- Model selection
- Cross-validation

**Usage**:
```python
from components.training.model_trainer import main
main()
```

### 5. Model Evaluation (`components/evaluation/`)

**Purpose**: Evaluates model performance and creates visualizations

**Features**:
- Comprehensive metrics
- Visualization plots
- Performance reports
- Threshold checking

**Usage**:
```python
from components.evaluation.model_evaluator import main
main()
```

## Deployment

### Kubernetes Deployment

1. **Apply persistent volume**:
   ```bash
   kubectl apply -f k8s/persistent-volume.yaml
   ```

2. **Deploy the pipeline**:
   ```bash
   kubectl apply -f k8s/kubeflow-pipeline.yaml
   ```

3. **Monitor the deployment**:
   ```bash
   kubectl get pods -n kubeflow
   kubectl logs -f <pod-name> -n kubeflow
   ```

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t ml-pipeline:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -v $(pwd):/workspace ml-pipeline:latest
   ```

## Monitoring

### Pipeline Monitoring

- **Kubeflow UI**: Monitor pipeline runs and logs
- **Kubernetes Dashboard**: Monitor pod status and resources
- **Custom Metrics**: Track model performance over time

### Model Monitoring

- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Data Drift**: Monitor feature distributions
- **Model Drift**: Track prediction distributions

### Logging

Logs are stored in the `logs/` directory with the following structure:
```
logs/
├── pipeline.log          # Main pipeline logs
├── data_fetching.log     # Data fetching logs
├── preprocessing.log      # Preprocessing logs
├── feature_engineering.log # Feature engineering logs
├── training.log          # Training logs
└── evaluation.log        # Evaluation logs
```

## Troubleshooting

### Common Issues

1. **Kubeflow not accessible**:
   ```bash
   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
   ```

2. **Pipeline fails to start**:
   - Check Kubernetes cluster status
   - Verify persistent volume claims
   - Check resource limits

3. **Component fails**:
   - Check component logs
   - Verify input data format
   - Check configuration parameters

4. **Model training issues**:
   - Verify data quality
   - Check hyperparameter ranges
   - Monitor resource usage

### Debugging

1. **Enable debug logging**:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check component outputs**:
   ```bash
   kubectl logs <pod-name> -n kubeflow
   ```

3. **Verify data flow**:
   - Check intermediate files in data directories
   - Verify file permissions
   - Monitor disk space

### Performance Optimization

1. **Resource allocation**:
   - Adjust CPU and memory limits
   - Use GPU resources for training
   - Optimize storage class

2. **Parallelization**:
   - Enable parallel model training
   - Use distributed computing
   - Optimize data loading

3. **Caching**:
   - Cache intermediate results
   - Use persistent storage
   - Implement result caching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs and documentation 