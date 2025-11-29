import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from common.schema import CATEGORIES

# Features that need special handling due to heavy right-skew
# Use QuantileTransformer which handles extreme values better than log transform
SKEWED_FEATURES = ['EncounterCount', 'MedicationCount', 'ConditionCount', 'TotalCost', 'ProcedureCount']

# Keep for backward compatibility
LOG_TRANSFORM_FEATURES = SKEWED_FEATURES

class DataPreprocessor:
    def __init__(self, continuous_cols=None, categorical_cols=None):
        self.continuous_cols = continuous_cols or []
        self.categorical_cols = categorical_cols or []
        self.pipeline = None
        self.feature_names_out = None
        # Use QuantileTransformer for ALL continuous features to handle non-Gaussian distributions
        # This is critical for features like BloodPressure and Glucose which have sharp peaks
        self.skewed_features = self.continuous_cols
        self.normal_features = [] # No features use StandardScaler anymore
        # Keep for backward compat
        self.log_features = self.skewed_features

    def fit(self, df):
        """
        Fits the preprocessor on the provided DataFrame.
        Uses predefined categories from schema to ensure consistent encoding.
        Uses QuantileTransformer for heavily-skewed features.
        """
        transformers = []
        
        # Handle normal continuous features with StandardScaler
        if self.normal_features:
            transformers.append(
                ('num_normal', StandardScaler(), self.normal_features)
            )
        
        # Handle skewed features with QuantileTransformer -> maps to normal distribution
        # This handles extreme values much better than log transform
        if self.skewed_features:
            quantile_transformer = QuantileTransformer(
                output_distribution='normal',  # Maps to standard normal
                n_quantiles=min(len(df), 500),  # Use all samples if <500
                random_state=42
            )
            transformers.append(
                ('num_skewed', quantile_transformer, self.skewed_features)
            )
        
        if self.categorical_cols:
            # Use predefined categories from schema for consistent encoding across all clients
            predefined_categories = [CATEGORIES.get(col, None) for col in self.categorical_cols]
            transformers.append(
                ('cat', OneHotEncoder(
                    categories=predefined_categories,
                    handle_unknown='ignore', 
                    sparse_output=False
                ), self.categorical_cols)
            )
            
        self.pipeline = ColumnTransformer(transformers=transformers)
        self.pipeline.fit(df)
        
        # Capture output feature names for reconstruction
        # Note: get_feature_names_out is available in scikit-learn >= 1.0
        try:
            self.feature_names_out = self.pipeline.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn or if not supported
            self.feature_names_out = []
            if self.continuous_cols:
                self.feature_names_out.extend(self.continuous_cols)
            if self.categorical_cols:
                # This is an approximation if we can't get exact names
                cat_encoder = self.pipeline.named_transformers_['cat']
                if hasattr(cat_encoder, 'categories_'):
                    for i, col in enumerate(self.categorical_cols):
                        for cat in cat_encoder.categories_[i]:
                            self.feature_names_out.append(f"{col}_{cat}")

    def transform(self, df):
        """
        Transforms the DataFrame into a numpy array (continuous + one-hot).
        """
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.pipeline.transform(df).astype(np.float32)

    def inverse_transform(self, data):
        """
        Reconstructs the DataFrame from the numpy array.
        Handles both normal continuous features and quantile-transformed features.
        """
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
            
        # Get transformers
        num_normal_trans = self.pipeline.named_transformers_.get('num_normal')
        num_skewed_trans = self.pipeline.named_transformers_.get('num_skewed')
        # Backward compatibility
        if num_skewed_trans is None:
            num_skewed_trans = self.pipeline.named_transformers_.get('num_log')
        cat_trans = self.pipeline.named_transformers_.get('cat')
        
        reconstructed = {}
        current_idx = 0
        
        # 1. Inverse Normal Continuous Features
        if self.normal_features and num_normal_trans:
            n_normal = len(self.normal_features)
            normal_data = data[:, current_idx : current_idx + n_normal]
            # Use wider clipping range to preserve variance in inverse transform
            normal_data = np.clip(normal_data, -5, 5)
            normal_original = num_normal_trans.inverse_transform(normal_data)
            
            for i, col in enumerate(self.normal_features):
                reconstructed[col] = normal_original[:, i]
            
            current_idx += n_normal
        
        # 2. Inverse Skewed (Quantile-Transformed) Features
        if self.skewed_features and num_skewed_trans:
            n_skewed = len(self.skewed_features)
            skewed_data = data[:, current_idx : current_idx + n_skewed]
            
            # Use wider clipping range to preserve variance
            # QuantileTransformer with normal output expects values roughly in [-3, 3]
            # but we allow [-4, 4] to capture more of the distribution
            skewed_data = np.clip(skewed_data, -4.0, 4.0)
            
            try:
                skewed_original = num_skewed_trans.inverse_transform(skewed_data)
            except Exception:
                # Fallback: assume it's a log pipeline
                scaler = num_skewed_trans.named_steps.get('scaler', num_skewed_trans)
                log_scaled = scaler.inverse_transform(skewed_data) if hasattr(scaler, 'inverse_transform') else skewed_data
                log_scaled = np.clip(log_scaled, 0, 12)
                skewed_original = np.expm1(log_scaled)
            
            for i, col in enumerate(self.skewed_features):
                # Ensure non-negative for count/cost features
                reconstructed[col] = np.maximum(0, skewed_original[:, i])
            
            current_idx += n_skewed
            
        # 3. Inverse Categorical
        if self.categorical_cols and cat_trans:
            n_cat_features = sum(len(cats) for cats in cat_trans.categories_)
            cat_data = data[:, current_idx : current_idx + n_cat_features]
            cat_original = cat_trans.inverse_transform(cat_data)
            
            for i, col in enumerate(self.categorical_cols):
                reconstructed[col] = cat_original[:, i]
                
            current_idx += n_cat_features
            
        return pd.DataFrame(reconstructed)

    def save(self, filepath):
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
