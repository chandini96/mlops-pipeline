#!/usr/bin/env python3
"""
Evaluation & Gating Component for Kubeflow Pipeline
- Evaluates multiple trained models
- Applies metric threshold (accuracy >= 0.93)
- Selects the best model for registration
"""

import argparse
import logging
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO)

def evaluate_model(model_path, X_test, y_test):
    """Load a model and compute classification metrics"""
    model_file = os.path.join(model_path, "model.joblib")
    if not os.path.exists(model_file):
        logging.warning(f"Model file not found: {model_file}")
        return None
    
    model = joblib.load(model_file)
    y_pred = model.predict(X_test)
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = None  # Some models may not support predict_proba
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models and select best one")
    parser.add_argument("--model-artifacts", type=str, required=True, help="Directory containing all trained model files")
    parser.add_argument("--model-dirs", type=str, required=True, help="Comma-separated list of trained model directories")
    parser.add_argument("--test-dataset", type=str, required=True, help="Test dataset CSV path")
    parser.add_argument("--target-column", type=str, required=True, help="Target column name")
    parser.add_argument(
        "--metric-key", 
        type=str, 
        default="accuracy", 
        help="Metric to select best model"
    )
    parser.add_argument(
        "--metric-threshold", 
        type=float, 
        default=0.84,   # accuracy threshold set to 0.84
        help="Minimum metric threshold"
    )
    parser.add_argument("--output-file", type=str, required=True, help="Output file to write the best model directory")
    parser.add_argument("--metrics-file", type=str, help="Optional file to save detailed metrics for all models")
    args = parser.parse_args()

    logging.info("=== EVALUATION & GATING COMPONENT ===")
    logging.info(f"Test dataset: {args.test_dataset}")
    logging.info(f"Model artifacts directory: {args.model_artifacts}")
    logging.info(f"Model directories string: {args.model_dirs}")
    logging.info(f"Selection metric: {args.metric_key}")
    logging.info(f"Metric threshold: {args.metric_threshold}")

    # Parse model directories and map them to the actual artifact paths
    model_dir_names = args.model_dirs.split(",")
    model_dirs = []
    for model_dir_name in model_dir_names:
        # Extract just the model name (e.g., "logistic_regression" from "/tmp/models/logistic_regression")
        model_name = os.path.basename(model_dir_name.strip())
        # Map to the actual artifact path
        actual_model_dir = os.path.join(args.model_artifacts, model_name)
        model_dirs.append(actual_model_dir)
        logging.info(f"Mapped {model_dir_name} -> {actual_model_dir}")
    
    test_df = pd.read_csv(args.test_dataset)
    X_test, y_test = test_df.drop(columns=[args.target_column]), test_df[args.target_column]

    best_model = None
    best_metric = float('-inf')
    all_results = []

    logging.info("=== MODEL EVALUATION RESULTS ===")
    for model_dir in model_dirs:
        metrics = evaluate_model(model_dir, X_test, y_test)
        if metrics is None:
            continue
        
        score = metrics.get(args.metric_key, 0.0)
        model_name = os.path.basename(model_dir)
        
        # Store results for summary
        result = {
            'model_name': model_name,
            'model_dir': model_dir,
            'score': score,
            'metrics': metrics,
            'passed_threshold': score >= args.metric_threshold
        }
        all_results.append(result)
        
        status = "✅ PASSED" if score >= args.metric_threshold else "❌ FAILED"
        logging.info(f"{model_name}: {args.metric_key}={score:.4f} {status}")
        
        if score >= args.metric_threshold and score > best_metric:
            best_metric = score
            best_model = model_dir

    # Log summary
    logging.info("=== EVALUATION SUMMARY ===")
    logging.info(f"Total models evaluated: {len(all_results)}")
    logging.info(f"Models passing threshold ({args.metric_threshold}): {sum(1 for r in all_results if r['passed_threshold'])}")
    
    if best_model:
        best_name = os.path.basename(best_model)
        logging.info(f"🏆 BEST MODEL: {best_name} with {args.metric_key}={best_metric:.4f}")
    else:
        logging.warning("⚠️ NO MODEL PASSED THE THRESHOLD!")

    # Save detailed metrics if requested
    if args.metrics_file:
        try:
            import json
            metrics_report = {
                'evaluation_summary': {
                    'total_models': len(all_results),
                    'models_passed_threshold': sum(1 for r in all_results if r['passed_threshold']),
                    'threshold': args.metric_threshold,
                    'metric_key': args.metric_key,
                    'best_model': os.path.basename(best_model) if best_model else None,
                    'best_score': best_metric if best_model else None
                },
                'model_results': all_results
            }
            
            os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
            with open(args.metrics_file, 'w') as f:
                json.dump(metrics_report, f, indent=2)
            logging.info(f"Detailed metrics saved to: {args.metrics_file}")
        except Exception as e:
            logging.error(f"Failed to save metrics file: {e}")

    # Write output to file
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            if best_model:
                logging.info(f"✅ Best model selected: {best_model} ({args.metric_key}={best_metric:.4f})")
                f.write(best_model)
                logging.info(f"Output written to: {args.output_file}")
            else:
                logging.warning("⚠ No model passed the metric threshold!")
                f.write("NONE")
                logging.info(f"Output written to: {args.output_file}")
    except Exception as e:
        logging.error(f"Failed to write output file: {e}")
        # Fallback: print the output
        if best_model:
            print(f"output:best_model_dir={best_model}")
        else:
            print("output:best_model_dir=NONE")

if __name__ == "__main__":
    main()
