#!/usr/bin/env python3
"""
Multi-Model Training Component for Kubeflow Pipeline
- Trains multiple classifiers: Logistic Regression, Decision Tree, Random Forest
- Saves each model in its own subdirectory inside --model-paths-dir
- Evaluation metrics are NOT computed here; they will be handled by a separate evaluation component
"""

import argparse
import logging
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)

def train_and_save_model(model, model_name, X_train, y_train, output_dir):
    """Train a model and save it to a specific folder"""
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_file = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_file)
    logging.info(f"{model_name} saved to {model_file}")
    
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Train multiple classification models")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--model-paths-dir", type=str, required=True, help="Base directory to save all trained models")
    parser.add_argument("--output-file", type=str, required=True, help="Output file to write the list of model directories")
    args = parser.parse_args()

    logging.info("=== MULTI-MODEL TRAINING COMPONENT ===")
    logging.info(f"Train dataset: {args.train_dataset}")
    logging.info(f"Test dataset: {args.test_dataset}")
    logging.info(f"Target column: {args.target_column}")
    logging.info(f"Models will be saved under: {args.model_paths_dir}")
    logging.info(f"Output file: {args.output_file}")

    # Load datasets
    train_df = pd.read_csv(args.train_dataset)
    X_train, y_train = train_df.drop(columns=[args.target_column]), train_df[args.target_column]

    # Define models to train
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(n_estimators=100),
    }

    # Use the model-paths-dir as the base directory for models
    base_output_dir = args.model_paths_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Train each model and save
    trained_model_dirs = []
    for name, model in models.items():
        model_dir = train_and_save_model(model, name, X_train, y_train, base_output_dir)
        trained_model_dirs.append(model_dir)

    # Write outputs for Kubeflow Pipelines (list of directories)
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            f.write(','.join(trained_model_dirs))
        logging.info(f"Output written to: {args.output_file}")
        logging.info(f"Model directories: {','.join(trained_model_dirs)}")
    except Exception as e:
        logging.error(f"Failed to write output file: {e}")
        # Fallback: print the output
        print(f"output:trained_model_dirs={','.join(trained_model_dirs)}")
    
    logging.info(" Training completed successfully for all models!")

if __name__ == "__main__":
    main()
