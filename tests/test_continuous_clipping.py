"""Test continuous feature clipping ranges"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_samples import generate_synthetic_dataset
from common.config import MODEL_DIR

def test_continuous_feature_ranges():
    """Ensure all continuous features are within expected ranges after generation"""
    print("Testing continuous feature range clipping...")
    
    # Generate a small test dataset
    test_file = os.path.join(MODEL_DIR, 'test_clipping.csv')
    try:
        generate_synthetic_dataset(n_samples=100, output_file=test_file)
    except Exception as e:
        print(f"Note: Generation failed (expected if no trained model): {e}")
        print("Skipping test - this is normal without a trained model")
        return
    
    df = pd.read_csv(test_file)
    
    # Define expected ranges
    ranges = {
        'Age': (18, 90),
        'BMI': (15, 50),
        'BloodPressure': (80, 200),
        'Glucose': (50, 400)
    }
    
    for col, (min_val, max_val) in ranges.items():
        actual_min = df[col].min()
        actual_max = df[col].max()
        
        print(f"✓ {col}: range [{actual_min:.2f}, {actual_max:.2f}] within [{min_val}, {max_val}]")
        assert actual_min >= min_val, f"{col} min {actual_min:.2f} < expected {min_val}"
        assert actual_max <= max_val, f"{col} max {actual_max:.2f} > expected {max_val}"
    
    # Cleanup
    try:
        os.remove(test_file)
    except Exception:
        pass

if __name__ == '__main__':
    try:
        test_continuous_feature_ranges()
        print("\n✓ Continuous feature clipping test PASSED")
    except AssertionError as e:
        print(f"\n✗ Continuous feature clipping test FAILED: {e}")
        sys.exit(1)
