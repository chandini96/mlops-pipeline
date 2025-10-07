#!/usr/bin/env python3
"""
Data Fetcher for GCS (Production-ready)
Downloads a CSV dataset from Google Cloud Storage and saves it locally
inside a container-writable directory. Kubeflow will capture
this local file as the output artifact.
"""

import argparse
import sys
import logging
import pandas as pd
import gcsfs
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def fetch_data(source_gcs_path: str, local_output_path: str):
    """
    Fetches a CSV dataset from GCS and saves it locally.

    Args:
        source_gcs_path (str): Source CSV file in GCS (gs://bucket/path/file.csv)
        local_output_path (str): Local output path for Kubeflow artifact
    """
    logger.info(f"Fetching dataset from: {source_gcs_path}")
    logger.info(f"Output path provided by Kubeflow: {local_output_path}")
    logger.info(f"Output path type: {type(local_output_path)}")
    logger.info(f"Output path length: {len(local_output_path) if isinstance(local_output_path, str) else 'N/A'}")
    
    # Validate output path
    if not local_output_path or not isinstance(local_output_path, str):
        raise ValueError(f"Invalid output path: {local_output_path}")
    
    # Normalize the path to handle any path separators
    local_output_path = os.path.normpath(local_output_path)
    logger.info(f"Normalized output path: {local_output_path}")
    
    fs = gcsfs.GCSFileSystem()

    try:
        # Read CSV from GCS
        with fs.open(source_gcs_path, "rb") as f:
            df = pd.read_csv(f)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Ensure output directory exists
        logger.info(f"Creating output directory for: {local_output_path}")
        output_dir = os.path.dirname(local_output_path)
        if output_dir:  # Only create directory if there's a path component
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory created/verified: {output_dir}")
        else:
            logger.info("Output is in current directory, no directory creation needed")

        # Save CSV locally (Kubeflow will capture this file)
        logger.info(f"Saving CSV to: {local_output_path}")
        try:
            df.to_csv(local_output_path, index=False)
            logger.info(f"Dataset saved locally to: {local_output_path}")
        except Exception as save_error:
            logger.warning(f"Failed to save to {local_output_path}: {save_error}")
            # Fallback: try to save to a simple filename in current directory
            fallback_path = "heart_disease_dataset.csv"
            logger.info(f"Trying fallback path: {fallback_path}")
            df.to_csv(fallback_path, index=False)
            logger.info(f"Dataset saved to fallback path: {fallback_path}")
            # Update the path for verification
            local_output_path = fallback_path

        # Verify file exists and get details
        if os.path.exists(local_output_path):
            file_size = os.path.getsize(local_output_path)
            logger.info(f"File exists at: {local_output_path}")
            logger.info(f"File size: {file_size} bytes")
            
            # List directory contents to see what's there
            dir_path = os.path.dirname(local_output_path)
            logger.info(f"Directory contents of {dir_path}:")
            try:
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        logger.info(f"   - {item} (file, {size} bytes)")
                    else:
                        logger.info(f"   - {item} (directory)")
            except Exception as e:
                logger.error(f"   Error listing directory: {e}")
        else:
            logger.error(f"File was not created at: {local_output_path}")
            raise FileNotFoundError(f"Output CSV was not created at {local_output_path}")

    except FileNotFoundError:
        logger.error(f"Source file not found: {source_gcs_path}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error while fetching data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Fetch dataset from GCS for Kubeflow.")
    parser.add_argument("--gcs-path", type=str, required=True, help="Source CSV path in GCS")
    parser.add_argument("--full-dataset", type=str, required=True, help="Local output path for Kubeflow artifact")
    args = parser.parse_args()

    logger.info(f"Arguments received: --gcs-path={args.gcs_path}, --full-dataset={args.full_dataset}")
    fetch_data(args.gcs_path, args.full_dataset)
    logger.info("Data fetching completed successfully!")

if __name__ == "__main__":
    main()
