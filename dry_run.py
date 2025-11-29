import torch
import pandas as pd
import numpy as np
from common.model import TabularDiffusionModel
from common.preprocessing import DataPreprocessor
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, get_input_dim
from common.data import generate_synthetic_data

def test_dry_run():
    print("Testing imports and initialization...")
    
    # 1. Test Data Generation
    df = generate_synthetic_data(10)
    print("Data generation successful. Shape:", df.shape)
    
    # 2. Test Preprocessor
    prep = DataPreprocessor(CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS)
    prep.fit(df)
    data_transformed = prep.transform(df)
    print("Preprocessing successful. Output shape:", data_transformed.shape)
    
    # 3. Test Model Initialization
    input_dim = get_input_dim()
    model = TabularDiffusionModel(input_dim=input_dim)
    print("Model initialization successful.")
    
    # 4. Test Forward Pass
    x = torch.randn(5, input_dim)
    t = torch.randint(0, 100, (5,))
    out = model(x, t)
    print("Forward pass successful. Output shape:", out.shape)
    
    print("\nALL CHECKS PASSED.")

if __name__ == "__main__":
    test_dry_run()
