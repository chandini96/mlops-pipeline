# Kubeflow ML Pipeline

This project demonstrates a complete Machine Learning pipeline using Kubeflow, covering the entire ML workflow from data fetching to model training and evaluation.

## Pipeline Components

1. **Data Fetching**: Downloads data from external sources or APIs
2. **Data Preprocessing**: Cleans and prepares the data for training
3. **Feature Engineering**: Creates and transforms features
4. **Model Training**: Trains ML models with hyperparameter tuning
5. **Model Evaluation**: Evaluates model performance
6. **Model Deployment**: Deploys the best model

## Project Structure

```
mlops-v2/
├── components/           # Pipeline components
│   ├── data_fetching/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── training/
│   ├── evaluation/
│   └── deployment/
├── pipeline/            # Main pipeline definition
├── data/               # Data storage
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks
└── config/            # Configuration files
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Kubeflow:
   ```bash
   # Install Kubeflow SDK
   pip install kfp
   
   # Set up Kubernetes cluster (if not already done)
   # Follow Kubeflow installation guide
   ```

3. Run the pipeline:
   ```bash
   python pipeline/main_pipeline.py
   ```

## Pipeline Features

- **Modular Design**: Each component is independent and reusable
- **Data Versioning**: Tracks data versions and lineage
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Model Registry**: Centralized model management
- **Monitoring**: Performance monitoring and alerting
- **Reproducibility**: Ensures reproducible ML experiments

## Configuration

Edit `config/pipeline_config.yaml` to customize:
- Data sources
- Model parameters
- Training configurations
- Deployment settings

## Monitoring

The pipeline includes:
- Model performance metrics
- Data quality checks
- Training logs
- Deployment status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License 