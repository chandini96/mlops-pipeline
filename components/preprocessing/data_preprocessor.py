import argparse
import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder
import sys

def load_dataset(path: str) -> pd.DataFrame:
    """Loads CSV dataset from local path or GCS (if gcsutil is configured)."""
    print(f"DEBUG: load_dataset called with path: '{path}'")
    print(f"DEBUG: Path type: {type(path)}")
    print(f"DEBUG: Path length: {len(path) if isinstance(path, str) else 'N/A'}")
    
    # Check if path looks like CSV content instead of a file path
    if isinstance(path, str) and len(path) > 1000 and ',' in path and '\n' in path:
        print("WARNING: Input appears to be CSV content instead of a file path!")
        print(f"First 200 characters: {path[:200]}...")
        print("This suggests the data fetching component didn't save the file properly")
        raise ValueError("Input appears to be CSV content instead of a file path. Check data fetching component.")
    
    # Check if file exists
    if os.path.exists(path):
        print(f"File exists at: {path}")
        file_size = os.path.getsize(path)
        print(f"File size: {file_size} bytes")
    else:
        print(f"File does not exist at: {path}")
        # List directory contents
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            print(f"Directory contents of {dir_path}:")
            try:
                for item in os.listdir(dir_path):
                    print(f"   - {item}")
            except Exception as e:
                print(f"   Error listing directory: {e}")
        else:
            print(f"Directory does not exist: {dir_path}")
    
    return pd.read_csv(path)

def save_dataframe(df: pd.DataFrame, path: str):
    """Saves dataframe as CSV to the given path."""
    try:
        # In Kubeflow, output paths are local paths that will be copied to GCS
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved dataframe to: {path}")
    except Exception as e:
        print(f"Error saving dataframe to {path}: {e}")
        raise

def save_json(obj: dict, path: str):
    """Saves Python dict as JSON."""
    # Ensure all values are JSON serializable by converting numpy types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        else:
            return obj
    
    # Convert numpy types before saving
    serializable_obj = convert_numpy_types(obj)
    
    try:
        # In Kubeflow, output paths are local paths that will be copied to GCS
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable_obj, f, indent=2)
        print(f"Saved JSON to: {path}")
    except Exception as e:
        print(f"Error saving JSON to {path}: {e}")
        raise

def preprocess(input_path: str, preprocessed_output: str, features_output: str, encoders_output: str):
    try:
        print(f"Loading dataset from {input_path}...")
        df = load_dataset(input_path)
        
        if df.empty:
            raise ValueError("Dataset is empty!")
        
        print(f"Loaded dataset with shape: {df.shape}")

        # Strip whitespace in column names
        df.columns = df.columns.str.strip()

        # Handle missing values safely
        for col in df.columns:
            try:
                if df[col].dtype == "object":  # categorical → fill with mode
                    mode_values = df[col].mode()
                    if len(mode_values) > 0:
                        df[col] = df[col].fillna(mode_values[0])
                    else:
                        df[col] = df[col].fillna("Unknown")
                else:  # numeric → fill with mean
                    mean_val = df[col].mean()
                    if pd.notna(mean_val):
                        df[col] = df[col].fillna(mean_val)
                    else:
                        df[col] = df[col].fillna(0)
            except Exception as e:
                print(f"Warning: Could not handle missing values in column {col}: {e}")
                # Fill with a safe default
                df[col] = df[col].fillna("Unknown" if df[col].dtype == "object" else 0)

        # Encode categorical columns (INCLUDE target column for proper encoding)
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        
        # Handle target column encoding separately
        target_encoder = None
        if "Heart Disease" in categorical_cols:
            print(f"Encoding target column 'Heart Disease'...")
            target_le = LabelEncoder()
            df["Heart Disease"] = target_le.fit_transform(df["Heart Disease"].astype(str))
            target_encoder = {
                "target_column": "Heart Disease",
                "classes": target_le.classes_.tolist(),
                "mapping": dict(zip(target_le.classes_.tolist(), target_le.transform(target_le.classes_).tolist())),
                "positive_class": 1,  # Assuming 'Presence' gets encoded as 1
                "negative_class": 0   # Assuming 'Absence' gets encoded as 0
            }
            print(f"Target column encoded. Classes: {target_le.classes_}")
            print(f"Target mapping: {target_encoder['mapping']}")
            categorical_cols.remove("Heart Disease")
        
        # Encode other categorical columns
        encoders = {}
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                # Convert numpy types to Python native types for JSON serialization
                encoded_mapping = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
                encoders[col] = encoded_mapping
                print(f"Encoded categorical column: {col}")
            except Exception as e:
                print(f"Warning: Could not encode column {col}: {e}")
                # Keep original values if encoding fails
                encoders[col] = {"error": "Encoding failed", "original_values": df[col].unique().tolist()}
        
        # Add target encoder to encoders dict
        if target_encoder:
            encoders["target_encoder"] = target_encoder

        # Extract features (exclude target if known)
        if "Heart Disease" in df.columns:
            feature_cols = [c for c in df.columns if c != "Heart Disease"]
            print(f"Target column 'Heart Disease' excluded from features. Feature count: {len(feature_cols)}")
        else:
            feature_cols = df.columns.tolist()
            print(f"All columns used as features. Feature count: {len(feature_cols)}")
        
        # Log target column information after encoding
        if "Heart Disease" in df.columns:
            target_values = df["Heart Disease"].unique()
            target_counts = df["Heart Disease"].value_counts().to_dict()
            print(f"Target column 'Heart Disease' encoded values: {target_values}")
            print(f"Target column value counts: {target_counts}")

        # Save outputs
        print(f"Saving preprocessed dataset to {preprocessed_output}...")
        save_dataframe(df, preprocessed_output)

        print(f"Saving features to {features_output}...")
        save_json({"features": feature_cols}, features_output)

        print(f"Saving encoders to {encoders_output}...")
        save_json(encoders, encoders_output)
        
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description="Preprocess heart disease dataset")
        parser.add_argument("--input_csv", type=str, required=True)
        parser.add_argument("--preprocessed_dataset_output", type=str, required=True)
        parser.add_argument("--features_output", type=str, required=True)
        parser.add_argument("--encoders_output", type=str, required=True)
        args = parser.parse_args()

        print(f"Starting preprocessing with arguments:")
        print(f"   Input: {args.input_csv}")
        print(f"   Output CSV: {args.preprocessed_dataset_output}")
        print(f"   Features: {args.features_output}")
        print(f"   Encoders: {args.encoders_output}")

        preprocess(
            args.input_csv,
            args.preprocessed_dataset_output,
            args.features_output,
            args.encoders_output,
        )
        
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
