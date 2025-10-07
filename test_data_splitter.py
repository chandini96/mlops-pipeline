#!/usr/bin/env python3
"""
Test script for the data splitter component
"""

import pandas as pd
import tempfile
import os
import sys

# Add the components directory to the path
sys.path.append('components/data_splitting')

# Import the data splitter functions
from data_splitter import load_data, split_data, save_to_local_temp

def test_data_splitter():
    """Test the data splitter component with sample data"""
    
    # Create sample data
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'Heart Disease': ['Absence', 'Presence', 'Absence', 'Presence', 'Absence', 
                         'Presence', 'Absence', 'Presence', 'Absence', 'Presence']
    }
    
    df = pd.DataFrame(data)
    print(f"Sample data shape: {df.shape}")
    print(f"Sample data columns: {list(df.columns)}")
    print(f"Sample data:\n{df.head()}")
    
    # Test split_data function
    print("\n--- Testing split_data function ---")
    train_df, test_df = split_data(df, 'Heart Disease', test_size=0.3, random_state=42)
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    
    # Test save_to_local_temp function
    print("\n--- Testing save_to_local_temp function ---")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, 'train.csv')
        test_path = os.path.join(temp_dir, 'test.csv')
        
        save_to_local_temp(train_df, train_path)
        save_to_local_temp(test_df, test_path)
        
        print(f"Training data saved to: {train_path}")
        print(f"Testing data saved to: {test_path}")
        
        # Verify files exist
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("All files created successfully!")
        else:
            print("Some files were not created!")
    
    print("\nData splitter test completed successfully!")

if __name__ == "__main__":
    test_data_splitter()
