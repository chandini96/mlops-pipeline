#!/usr/bin/env python3
"""
Setup script for Kubeflow ML Pipeline
Handles installation and configuration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    try:
        directories = [
            "data/raw",
            "data/processed", 
            "data/engineered",
            "models",
            "evaluation",
            "logs",
            "notebooks"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

def check_kubeflow_installation():
    """Check if Kubeflow is installed"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if kubectl is available
        result = subprocess.run(["kubectl", "version", "--client"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("kubectl not found. Please install kubectl first.")
            return False
        
        # Check if Kubeflow namespace exists
        result = subprocess.run(["kubectl", "get", "namespace", "kubeflow"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Kubeflow namespace not found. Please install Kubeflow first.")
            return False
        
        logger.info("Kubeflow installation detected")
        return True
        
    except FileNotFoundError:
        logger.warning("kubectl not found. Please install kubectl first.")
        return False
    except Exception as e:
        logger.error(f"Error checking Kubeflow installation: {e}")
        return False

def build_docker_image():
    """Build Docker image for the pipeline"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Building Docker image...")
        subprocess.check_call(["docker", "build", "-t", "ml-pipeline:v1.0.0", "."])
        logger.info("Docker image built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building Docker image: {e}")
        return False
    except FileNotFoundError:
        logger.warning("Docker not found. Please install Docker first.")
        return False

def create_persistent_volume():
    """Create persistent volume for pipeline storage"""
    logger = logging.getLogger(__name__)
    
    try:
        pv_yaml = """
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ml-pipeline-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /tmp/ml-pipeline
  storageClassName: standard
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-pipeline-workspace
  namespace: kubeflow
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
"""
        
        with open("k8s/persistent-volume.yaml", "w") as f:
            f.write(pv_yaml)
        
        logger.info("Persistent volume YAML created")
        return True
        
    except Exception as e:
        logger.error(f"Error creating persistent volume: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Running basic tests...")
        
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        import kfp
        
        logger.info("All imports successful")
        
        # Test configuration loading
        import yaml
        with open("config/pipeline_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loading successful")
        
        logger.info("All tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger = setup_logging()
    
    logger.info("Starting Kubeflow ML Pipeline setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Check Kubeflow installation
    kubeflow_installed = check_kubeflow_installation()
    if not kubeflow_installed:
        logger.warning("Kubeflow not detected. Please install Kubeflow to run the pipeline.")
    
    # Build Docker image
    docker_built = build_docker_image()
    if not docker_built:
        logger.warning("Docker image build failed. Pipeline may not work correctly.")
    
    # Create persistent volume
    if not create_persistent_volume():
        logger.warning("Failed to create persistent volume configuration.")
    
    # Run tests
    if not run_tests():
        logger.error("Tests failed. Please check the installation.")
        sys.exit(1)
    
    logger.info("Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Install Kubeflow if not already installed")
    logger.info("2. Apply persistent volume: kubectl apply -f k8s/persistent-volume.yaml")
    logger.info("3. Run the pipeline: python pipeline/main_pipeline.py")
    logger.info("4. Or deploy to Kubernetes: kubectl apply -f k8s/kubeflow-pipeline.yaml")

if __name__ == "__main__":
    main() 