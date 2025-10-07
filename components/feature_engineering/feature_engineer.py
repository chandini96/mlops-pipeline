#!/usr/bin/env python3
"""
Feature Engineering Script
Performs scaling on the dataset using StandardScaler.
Supports local and GCS paths for input/output.
"""

import argparse
import logging
import pandas as pd
import gcsfs
import joblib
from sklearn.preprocessing import StandardScaler
import os
from typing import Tuple
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_path: str) -> pd.DataFrame:
    """Load data from GCS or local path"""
    logger.info(f"Loading dataset from: {input_path}")
    if input_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(input_path, 'r') as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(input_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {list(df.columns)}")
    return df

def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler"""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    logger.info(f"Numeric columns to scale: {list(numeric_cols)}")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    logger.info("Scaling completed.")
    return X_scaled, scaler

def save_to_path(df: pd.DataFrame, scaler: StandardScaler, output_csv_path: str, scaler_path: str):
    """Save DataFrame and scaler to local or GCS paths"""
    try:
        # Save DataFrame
        if output_csv_path.startswith("gs://"):
            logger.info(f"Saving scaled dataset to GCS: {output_csv_path}")
            df.to_csv(output_csv_path, index=False, storage_options={"token": "cloud"})
        else:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved scaled dataset locally: {output_csv_path}")

        # Save scaler (always local first, then user can upload to GCS if needed)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            joblib.dump(scaler, f)
        logger.info(f"Scaler saved locally: {scaler_path}")

    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Feature Engineering - Scale features for ML training')
    parser.add_argument('--input-dataset', type=str, required=True,
                        help='GCS or local path to input CSV file')
    parser.add_argument('--target-column', type=str, required=True,
                        help='Name of the target column')
    parser.add_argument('--output-dataset', type=str, required=True,
                        help='Output path for the scaled dataset')
    parser.add_argument('--scaler-path', type=str, required=True,
                        help='Output path for the saved scaler')
    args = parser.parse_args()

    try:
        logger.info("Starting feature engineering...")

        df = load_data(args.input_dataset)

        # Check target column
        if args.target_column not in df.columns:
            raise ValueError(f"Target column '{args.target_column}' not found in dataset! Available columns: {list(df.columns)}")

        y = df[args.target_column]
        X = df.drop(columns=[args.target_column])

        X_scaled, scaler = scale_features(X)

        df_scaled = X_scaled.copy()
        df_scaled[args.target_column] = y.reset_index(drop=True)

        save_to_path(df_scaled, scaler, args.output_dataset, args.scaler_path)

        logger.info("Feature engineering completed successfully!")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
