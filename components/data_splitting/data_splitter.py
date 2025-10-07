#!/usr/bin/env python3
"""
Data Splitting Component for Kubeflow Pipeline
Splits dataset into train and test sets for model training
"""

import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_path: str) -> pd.DataFrame:
    """Load data from local path or GCS path"""
    logger.info(f"Loading dataset from: {input_path}")
    
    try:
        if input_path.startswith("gs://"):
            # Handle GCS path
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            with fs.open(input_path, 'r') as f:
                df = pd.read_csv(f)
        else:
            # Handle local path
            df = pd.read_csv(input_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {input_path}: {e}")
        raise

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets"""
    logger.info("Splitting data into train and test sets...")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Testing set shape: {test_df.shape}")

    return train_df, test_df

def save_to_local_temp(df: pd.DataFrame, output_path: str):
    """Save DataFrame to a local path, ensuring the directory exists"""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created/verified output directory: {output_dir}")
        
        # Save the DataFrame
        df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to: {output_path}")
        
        # Verify the file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"File created successfully: {output_path} (size: {file_size} bytes)")
        else:
            logger.error(f"File was not created at: {output_path}")
            raise FileNotFoundError(f"Output file was not created at {output_path}")
            
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Data Splitting - Split dataset into train and test sets')
    parser.add_argument('--input-dataset', type=str, required=True,
                        help='Path to input dataset (from Kubeflow Dataset artifact)')
    parser.add_argument('--target-column', type=str, required=True,
                        help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--train-path', type=str, required=True,
                        help='Local path for the training dataset (Kubeflow outputPath)')
    parser.add_argument('--test-path', type=str, required=True,
                        help='Local path for the testing dataset (Kubeflow outputPath)')
    args = parser.parse_args()

    try:
        logger.info("Starting data splitting...")
        logger.info(f"Arguments received:")
        logger.info(f"  - input_dataset: {args.input_dataset}")
        logger.info(f"  - target_column: {args.target_column}")
        logger.info(f"  - test_size: {args.test_size}")
        logger.info(f"  - random_state: {args.random_state}")
        logger.info(f"  - train_path: {args.train_path}")
        logger.info(f"  - test_path: {args.test_path}")

        df = load_data(args.input_dataset)

        if args.target_column not in df.columns:
            logger.error(f"Target column '{args.target_column}' not found in dataset!")
            logger.error(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Target column '{args.target_column}' not found. Available columns: {list(df.columns)}")

        logger.info(f"Target column '{args.target_column}' found successfully")

        train_df, test_df = split_data(
            df, args.target_column, args.test_size, args.random_state
        )

        # Save datasets locally for Kubeflow outputs
        logger.info("Saving training dataset...")
        save_to_local_temp(train_df, args.train_path)
        logger.info("Saving testing dataset...")
        save_to_local_temp(test_df, args.test_path)

        logger.info("Data splitting completed successfully!")

    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
