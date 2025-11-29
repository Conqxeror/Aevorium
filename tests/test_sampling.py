import os
import pandas as pd
import numpy as np
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, get_input_dim
from generate_samples import generate_synthetic_dataset, postprocess_synthetic_array
from common.data import generate_synthetic_data
from common.preprocessing import DataPreprocessor


def test_categorical_distributions_have_multiple_categories():
    # Fit a preprocessor from real data
    ref = generate_synthetic_data(n_samples=500)
    preprocessor = DataPreprocessor(continuous_cols=CONTINUOUS_COLUMNS, categorical_cols=CATEGORICAL_COLUMNS)
    preprocessor.fit(ref)

    # Create a random synthetic array with appropriate input_dim
    input_dim = get_input_dim()
    n_samples = 500
    # random floats simulating raw continuous + categorical outputs from the diffusion model
    synthetic_arr = np.random.normal(size=(n_samples, input_dim)).astype(np.float32)

    df = postprocess_synthetic_array(preprocessor, synthetic_arr)

    for col in CATEGORICAL_COLUMNS:
        assert col in df.columns, f"Column {col} missing from generated data"
        unique_count = df[col].nunique()
        # Ensure not all values are collapsed to a single category
        assert unique_count > 1, f"Categorical column {col} collapsed to {unique_count} unique value(s)"


def test_continuous_columns_finite_values():
    ref = generate_synthetic_data(n_samples=200)
    preprocessor = DataPreprocessor(continuous_cols=CONTINUOUS_COLUMNS, categorical_cols=CATEGORICAL_COLUMNS)
    preprocessor.fit(ref)
    input_dim = get_input_dim()
    synthetic_arr = np.random.normal(size=(200, input_dim)).astype(np.float32)
    df = postprocess_synthetic_array(preprocessor, synthetic_arr)
    for col in CONTINUOUS_COLUMNS:
        assert col in df.columns
        assert np.isfinite(df[col]).all(), f"Non-finite values found in {col}"
