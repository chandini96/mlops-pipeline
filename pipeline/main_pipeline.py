#!/usr/bin/env python3
"""
Production-ready Kubeflow + Vertex AI MLOps pipeline
- Uses custom Docker images for all components
- Tracks runs with Vertex AI Experiments
- Registers trained model in Vertex AI Model Registry
"""

import kfp
from kfp import dsl, compiler, components
from google.cloud import aiplatform
from datetime import datetime
import logging
import os

# -------------------------
# Constants
# -------------------------
PROJECT_ID = "pro-vigil-dev"
REGION = "us-central1"
# PIPELINE_ROOT = "gs://mlops-vl2/pipeline-root"  # Commented out to use Vertex AI default
MODEL_DISPLAY_NAME = "heart-disease-mlops-model"
EXPERIMENT_NAME = "heart-disease-classification-experiment"

# Component root path - robust absolute path handling
COMPONENT_ROOT = os.path.join(os.path.dirname(__file__), "..", "components")

logging.basicConfig(level=logging.INFO)


# -------------------------
# Pipeline Definition
# -------------------------
@dsl.pipeline(
    name="mlops-full-pipeline",
    description="Complete MLOps pipeline: Data fetching, preprocessing, feature engineering, splitting, training, evaluation, and model registry."
)
def full_mlops_pipeline_custom_images(
    gcs_data_path: str = "gs://mlops-vl2/Heart_Disease_Prediction.csv",
    target_column: str = "Heart Disease",
    model_type: str = "logistic_regression"
):
    """Kubeflow Pipeline DAG"""

    # Load components (Docker images are defined in each YAML) - using absolute paths
    data_fetch_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "data_fetching/component.yaml")
    )
    preprocessing_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "preprocessing/component.yaml")
    )
    feature_engineering_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "feature_engineering/component.yaml")
    )
    data_split_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "data_splitting/component.yaml")
    )
    train_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "training/multi_model_training.yaml")
    )
    evaluation_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "model_evaluation/evaluation.yaml")
    )
    model_registry_op = components.load_component_from_file(
        os.path.join(COMPONENT_ROOT, "model_registry/component.yaml")
    )

    # --- Step 1: Data Fetching ---
    fetch_task = data_fetch_op(
        gcs_path=gcs_data_path
    )
    fetch_task.set_cpu_limit("4").set_memory_limit("8G")

    # --- Step 2: Preprocessing ---
    preprocess_task = preprocessing_op(
        input_dataset=fetch_task.outputs["full_dataset"]
    )
    preprocess_task.set_cpu_limit("4").set_memory_limit("8G")

    # --- Step 3: Feature Engineering ---
    feature_task = feature_engineering_op(
        input_dataset=preprocess_task.outputs["preprocessed_dataset"],
        target_column=target_column
    )
    feature_task.set_cpu_limit("4").set_memory_limit("8G")

    # --- Step 4: Data Splitting ---
    split_task = data_split_op(
        input_dataset=feature_task.outputs["scaled_dataset"],
        target_column=target_column
    )
    split_task.set_cpu_limit("4").set_memory_limit("8G")

    # --- Step 5: Model Training ---
    train_task = train_op(
        train_dataset=split_task.outputs["train_dataset"],
        test_dataset=split_task.outputs["test_dataset"],
        target_column=target_column
    )
    train_task.set_cpu_limit("8").set_memory_limit("16G")

    # --- Step 6: Model Evaluation ---
    evaluation_task = evaluation_op(
        model_artifacts=train_task.outputs["model_artifacts"],
        model_dirs=train_task.outputs["trained_model_dirs"],
        test_dataset=split_task.outputs["test_dataset"],
        target_column=target_column,
        metric_key="accuracy",
        metric_threshold=0.84
    )
    evaluation_task.set_cpu_limit("4").set_memory_limit("8G")
    evaluation_task.after(train_task)

    # --- Step 7: Model Registry ---
    registry_task = model_registry_op(
        action="register",
        registry_path="gs://mlops-vl2/model-registry",
        model_name="heart-disease-classifier",
        model_path=evaluation_task.outputs["best_model_dir"],
        model_type=model_type,
        dataset_info=f"Dataset: {gcs_data_path}, Target: {target_column}",
        version="",  # Auto-generated
        project=PROJECT_ID,
        region=REGION,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    registry_task.set_cpu_limit("1").set_memory_limit("2G")
    registry_task.after(evaluation_task)  # Ensure evaluation completes before registry


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    logging.info("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=full_mlops_pipeline_custom_images,
        package_path="mlops_full_pipeline.yaml"
    )
    logging.info("Pipeline compiled successfully.")

    logging.info("Initializing Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)

    parameter_values = {
        "gcs_data_path": "gs://mlops-vl2/Heart_Disease_Prediction.csv",
        "target_column": "Heart Disease",
        "model_type": "logistic_regression",
    }

    job_id = f"mlops-full-pipeline-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    logging.info(f"Submitting pipeline job: {job_id}")

    pipeline_job = aiplatform.PipelineJob(
        display_name="mlops-full-pipeline-run",
        template_path="mlops_full_pipeline.yaml",
        # pipeline_root=PIPELINE_ROOT,  # Commented out to use Vertex AI default
        parameter_values=parameter_values,
        job_id=job_id,
        enable_caching=False,  # Disable caching for debugging
    )

    pipeline_job.run(sync=True)  # Wait until completion

    logging.info("Pipeline job submitted successfully.")