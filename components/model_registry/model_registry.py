#!/usr/bin/env python3
"""
Model Registry Component for Kubeflow Pipeline
- Stores registry.json in GCS (custom metadata)
- Uploads selected model to Vertex AI Model Registry
"""

import argparse
import logging
import json
import gcsfs
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Model Registry for managing model versions and metadata"""

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.fs = gcsfs.GCSFileSystem()

    def load_registry(self) -> Dict[str, Any]:
        try:
            with self.fs.open(f"{self.registry_path}/registry.json", 'r') as f:
                return json.load(f)
        except Exception:
            return {"models": {}, "latest_versions": {}}

    def save_registry(self, registry: Dict[str, Any]):
        with self.fs.open(f"{self.registry_path}/registry.json", 'w') as f:
            json.dump(registry, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        dataset_info: str,
        version: Optional[str] = None,
        project: Optional[str] = None,
        region: Optional[str] = None,
        serving_container_image_uri: Optional[str] = None,
    ) -> str:
        """Register the selected model version locally + Vertex AI"""

        if model_path == "NONE":
            logger.warning("No model selected by evaluation. Skipping registration.")
            return "NONE"

        logger.info(f"Registering model: {model_name}")
        logger.info(f"Model path: {model_path}")

        registry = self.load_registry()

        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"

        # Determine model file path
        model_file_path = None
        model_exists = False
        try:
            candidate_files = ["model.pkl", "model.joblib"]
            for f in candidate_files:
                path = os.path.join(model_path, f) if not model_path.startswith("gs://") else f"{model_path}/{f}"
                if os.path.exists(path) if not model_path.startswith("gs://") else self.fs.exists(path):
                    model_file_path = path
                    model_exists = True
                    break
        except Exception as e:
            logger.warning(f"Could not verify model file existence: {e}")

        if not model_exists:
            logger.error(f"Model file not found in {model_path}. Expected model.pkl or model.joblib")
            raise FileNotFoundError(f"Model file not found in {model_path}. Expected model.pkl or model.joblib")

        # Build registry entry
        model_entry = {
            "model_name": model_name,
            "model_type": model_type,
            "version": version,
            "model_path": model_path,
            "model_file_path": model_file_path,
            "dataset_info": dataset_info,
            "registration_date": datetime.now().isoformat(),
            "status": "active",
            "model_exists": model_exists,
        }

        # Update registry JSON
        if model_name not in registry["models"]:
            registry["models"][model_name] = {}
        registry["models"][model_name][version] = model_entry
        registry["latest_versions"][model_name] = version
        self.save_registry(registry)
        logger.info(f"Model {model_name} version {version} saved in GCS registry")

        # Vertex AI upload
        if project and region and serving_container_image_uri:
            try:
                logger.info("Uploading model to Vertex AI Model Registry...")
                aiplatform.init(project=project, location=region)
                artifact_uri = model_path if model_path.startswith("gs://") else model_path
                if not artifact_uri.endswith("/"):
                    artifact_uri += "/"
                uploaded_model = aiplatform.Model.upload(
                    display_name=model_name,
                    artifact_uri=artifact_uri,
                    serving_container_image_uri=serving_container_image_uri,
                )
                logger.info(f"Model uploaded to Vertex AI: {uploaded_model.resource_name}")
            except Exception as e:
                logger.error(f"Vertex AI upload failed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("Continuing without Vertex AI upload...")

        return version

def main():
    parser = argparse.ArgumentParser(description='Model Registry - Manage model versions and metadata')
    parser.add_argument('--action', type=str, required=True, choices=['register'])
    parser.add_argument('--registry-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True, help="Path from Evaluation component")
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--dataset-info', type=str, required=True)
    parser.add_argument('--version', type=str)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--serving-container-image-uri', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True, help="Output file to write the registry result")

    args = parser.parse_args()

    registry = ModelRegistry(args.registry_path)

    if args.action == 'register':
        try:
            version = registry.register_model(
                model_name=args.model_name,
                model_path=args.model_path,
                model_type=args.model_type,
                dataset_info=args.dataset_info,
                version=args.version,
                project=args.project,
                region=args.region,
                serving_container_image_uri=args.serving_container_image_uri,
            )
            logger.info(f"Model registered with version: {version}")
            
            # Write output to file
            try:
                os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
                with open(args.output_file, 'w') as f:
                    f.write(version)
                logger.info(f"Output written to: {args.output_file}")
            except Exception as write_error:
                logger.error(f"Failed to write output file: {write_error}")
                # Fallback: print the output
                print(f"registry_output={version}")
                
        except Exception as e:
            logger.error(f"Error in model registration: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Write error to file
            try:
                os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
                with open(args.output_file, 'w') as f:
                    f.write(f"ERROR: {e}")
                logger.info(f"Error output written to: {args.output_file}")
            except Exception as write_error:
                logger.error(f"Failed to write error output file: {write_error}")
                # Fallback: print the output
                print(f"registry_output=ERROR: {e}")
            raise

if __name__ == "__main__":
    main()
