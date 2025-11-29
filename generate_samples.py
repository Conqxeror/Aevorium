import torch
import numpy as np
import pandas as pd
import sys
import os
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.model import TabularDiffusionModel, DiffusionManager
from common.data import generate_synthetic_data
from common.security import decrypt_file
import io
from common.config import MODEL_DIR, SYNTHETIC_DATA_FILE

def load_latest_model_weights():
    # Find the latest global model file
    files = glob.glob(os.path.join(MODEL_DIR, "global_model_round_*.npz"))
    if not files:
        raise FileNotFoundError("No global model file found. Run training first.")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading weights from {latest_file}...")
    
    # Decrypt data
    decrypted_bytes = decrypt_file(latest_file)
    
    # Load from bytes
    data = np.load(io.BytesIO(decrypted_bytes))
    
    # Check if saved with names or positional
    file_keys = list(data.files)
    if file_keys and not file_keys[0].startswith('arr_'):
        # Named save: return dict keyed by param names
        return {k: data[k] for k in file_keys}
    else:
        # Legacy positional save: return list
        weights = [data[f"arr_{i}"] for i in range(len(file_keys))]
        return weights

def denormalize(data, min_val, max_val):
    # data is in [-1, 1]
    # x_norm = 2 * (x - min) / (max - min) - 1
    # x = (x_norm + 1) / 2 * (max - min) + min
    return (data + 1) / 2 * (max_val - min_val) + min_val

from common.preprocessing import DataPreprocessor, LOG_TRANSFORM_FEATURES
from common.config import MODEL_DIR
import joblib
import math
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, get_input_dim, FEATURE_RANGES, CATEGORIES
from common.data import load_real_healthcare_data

# Cache for real data statistics
REAL_DATA_STATS = None

def get_real_data_stats():
    """Load and cache real data statistics for distribution matching."""
    global REAL_DATA_STATS
    if REAL_DATA_STATS is None:
        try:
            real_df = load_real_healthcare_data()
            REAL_DATA_STATS = {
                'mean': {},
                'std': {},
                'percentiles': {},
                'correlations': {}
            }
            for col in CONTINUOUS_COLUMNS:
                if col in real_df.columns:
                    REAL_DATA_STATS['mean'][col] = real_df[col].mean()
                    REAL_DATA_STATS['std'][col] = real_df[col].std()
                    REAL_DATA_STATS['percentiles'][col] = {
                        '5': real_df[col].quantile(0.05),
                        '95': real_df[col].quantile(0.95)
                    }
            # Store correlation matrix for correlation-aware post-processing
            cont_cols = [c for c in CONTINUOUS_COLUMNS if c in real_df.columns]
            if cont_cols:
                REAL_DATA_STATS['correlations'] = real_df[cont_cols].corr().to_dict()
        except Exception as e:
            print(f"Warning: Could not load real data statistics: {e}")
            REAL_DATA_STATS = {'mean': {}, 'std': {}, 'percentiles': {}, 'correlations': {}}
    return REAL_DATA_STATS


def apply_correlation_adjustments(df, real_stats):
    """
    Apply lightweight correlation-aware adjustments to improve feature relationships.
    
    Focus on the most important correlations:
    - EncounterCount <-> TotalCost (strong positive ~0.8+)
    - ConditionCount <-> TotalCost (moderate positive ~0.6)
    - Age <-> BMI (moderate positive ~0.5)
    - Age <-> ConditionCount (moderate positive ~0.4)
    """
    real_corr = real_stats.get('correlations', {})
    if not real_corr:
        return df
    
    # Key correlation pairs to reinforce
    # Format: (driver_col, dependent_col, target_correlation_range)
    correlation_pairs = [
        ('EncounterCount', 'TotalCost', 0.6),
        ('ConditionCount', 'EncounterCount', 0.5),
        ('ConditionCount', 'TotalCost', 0.5),
        ('MedicationCount', 'TotalCost', 0.5),
        ('Age', 'BMI', 0.4),
        ('Age', 'ConditionCount', 0.3),
        ('Age', 'MedicationCount', 0.3),
    ]
    
    for driver_col, dependent_col, min_corr in correlation_pairs:
        if driver_col not in df.columns or dependent_col not in df.columns:
            continue
        
        # Check current synthetic correlation
        current_corr = df[[driver_col, dependent_col]].corr().iloc[0, 1]
        
        # Get target correlation from real data
        target_corr = real_corr.get(driver_col, {}).get(dependent_col, min_corr)
        
        # Only adjust if correlation is too weak
        if abs(current_corr) < abs(target_corr) * 0.5:
            # Blend dependent column with driver column to strengthen correlation
            # Use small blend factor to avoid distorting distribution too much
            blend_factor = 0.15
            
            # Normalize both columns
            driver_norm = (df[driver_col] - df[driver_col].mean()) / (df[driver_col].std() + 1e-8)
            dep_mean = df[dependent_col].mean()
            dep_std = df[dependent_col].std()
            
            # Blend while preserving the dependent column's distribution
            df[dependent_col] = (
                df[dependent_col] * (1 - blend_factor) + 
                (driver_norm * dep_std + dep_mean) * blend_factor
            )
    
    return df

# Real data distributions for categorical features (computed once)
REAL_CATEGORICAL_DISTRIBUTIONS = None

def get_real_categorical_distributions():
    """Load and cache the real data categorical distributions for distribution matching."""
    global REAL_CATEGORICAL_DISTRIBUTIONS
    if REAL_CATEGORICAL_DISTRIBUTIONS is None:
        try:
            real_df = load_real_healthcare_data()
            REAL_CATEGORICAL_DISTRIBUTIONS = {}
            for col in CATEGORICAL_COLUMNS:
                if col in real_df.columns:
                    dist = real_df[col].value_counts(normalize=True)
                    REAL_CATEGORICAL_DISTRIBUTIONS[col] = dist.to_dict()
        except Exception as e:
            print(f"Warning: Could not load real data distributions: {e}")
            REAL_CATEGORICAL_DISTRIBUTIONS = {}
    return REAL_CATEGORICAL_DISTRIBUTIONS


def postprocess_synthetic_array(preprocessor: DataPreprocessor, synthetic_arr: np.ndarray, deterministic: bool = False, temperature: float = 0.5, use_mode_for_rare: bool = True, match_real_distribution: bool = True) -> pd.DataFrame:
    """
    Convert raw model outputs (continuous + categorical continuous floats) to a DataFrame by
    converting per-category group floats into categorical one-hot vectors (via softmax + sampling),
    and then calling preprocessor.inverse_transform.
    
    Args:
        temperature: Lower values (e.g., 0.3-0.5) make sampling more deterministic,
                    higher values (e.g., 1.0+) make it more random. Default 0.5 for sharper distributions.
        use_mode_for_rare: If True, use mode (argmax) for categories where one option dominates >90%
        match_real_distribution: If True, sample categoricals from real data distributions
    """
    cat_trans_local = preprocessor.pipeline.named_transformers_.get('cat') if hasattr(preprocessor, 'pipeline') else None
    if cat_trans_local is None or not hasattr(cat_trans_local, 'categories_') or len(preprocessor.categorical_cols) == 0:
        return preprocessor.inverse_transform(synthetic_arr)

    # Get number of normal and log features
    n_normal = len(preprocessor.normal_features) if hasattr(preprocessor, 'normal_features') else 0
    n_log = len(preprocessor.log_features) if hasattr(preprocessor, 'log_features') else 0
    n_cont_local = n_normal + n_log
    
    # Fallback for old preprocessor format
    if n_cont_local == 0:
        n_cont_local = len(CONTINUOUS_COLUMNS)
    
    cat_sizes_local = [len(cats) for cats in cat_trans_local.categories_]
    n_cat_features_local = sum(cat_sizes_local)
    if synthetic_arr.shape[1] < n_cont_local + n_cat_features_local:
        raise ValueError("Synthetic data has fewer columns than expected by preprocessor")

    cont_data_local = synthetic_arr[:, :n_cont_local]
    cat_raw_local = synthetic_arr[:, n_cont_local:n_cont_local + n_cat_features_local]
    
    # Get real distributions if matching is enabled
    real_dists = get_real_categorical_distributions() if match_real_distribution else {}
    
    cat_onehot_parts_local = []
    idx_local = 0
    n_samples = synthetic_arr.shape[0]
    
    for k_idx, k_local in enumerate(cat_sizes_local):
        col_name = preprocessor.categorical_cols[k_idx]
        categories = list(cat_trans_local.categories_[k_idx])
        
        if match_real_distribution and col_name in real_dists:
            # Use real data distribution for sampling
            real_dist = real_dists[col_name]
            probs = np.array([real_dist.get(cat, 0.0) for cat in categories])
            probs = probs / probs.sum()  # Normalize
            draws_local = np.random.choice(k_local, size=n_samples, p=probs)
        else:
            # Fallback to model-based softmax sampling
            group_raw_local = cat_raw_local[:, idx_local: idx_local + k_local]
            scaled_logits = group_raw_local / max(temperature, 0.01)
            group_exp_local = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs_local = group_exp_local / np.sum(group_exp_local, axis=1, keepdims=True)
            
            if deterministic:
                draws_local = np.argmax(probs_local, axis=1)
            else:
                draws_local = np.array([np.random.choice(k_local, p=probs_local[i]) for i in range(n_samples)])
        
        onehot_local = np.eye(k_local)[draws_local]
        cat_onehot_parts_local.append(onehot_local.astype(np.float32))
        idx_local += k_local
    
    cat_onehot_final_local = np.concatenate(cat_onehot_parts_local, axis=1)
    synthetic_post_local = np.concatenate([cont_data_local, cat_onehot_final_local], axis=1)
    return preprocessor.inverse_transform(synthetic_post_local)

def generate_synthetic_dataset(n_samples=1000, output_file=SYNTHETIC_DATA_FILE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model
    input_dim = get_input_dim()
    model = TabularDiffusionModel(input_dim=input_dim)
    manager = DiffusionManager(model, device=device)
    
    # 2. Load Weights
    try:
        weights = load_latest_model_weights()
        if isinstance(weights, dict):
            # Named save: load directly
            state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        else:
            # Positional save: zip with keys
            params_dict = zip(model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: could not load model weights: {e}. Proceeding with randomly initialized model for sampling.")

    # 3. Generate Samples (in latent/transformed space)
    # Use DDIM for stable sampling with balanced stochasticity
    # eta=0.5 provides better diversity while maintaining quality
    # 300 steps improves sample quality at minor speed cost
    print(f"Generating {n_samples} synthetic samples using DDIM...")
    synthetic_tensor = manager.sample(n_samples, input_dim=input_dim, use_ddim=True, ddim_steps=300, eta=0.5)
    synthetic_data = synthetic_tensor.cpu().numpy()
    
    # Clip latent space outputs with wider range to preserve variance
    # QuantileTransformer can handle values in [-5, 5] for inverse_transform
    synthetic_data = np.clip(synthetic_data, -4.5, 4.5)
    
    # 4. Inverse Transform
    # Attempt to load a preprocessor (persisted during training). If not found, fallback to fitting a new one
    print("Loading or fitting preprocessor for inverse_transform...")
    preprocessor = None
    # Check for canonical preprocessor saved by clients/servers
    canonical_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if os.path.exists(canonical_path):
        try:
            preprocessor = DataPreprocessor.load(canonical_path)
            print(f"Loaded preprocessor from {canonical_path}")
        except Exception as e:
            print(f"Failed to load canonical preprocessor: {e}")

    # If no preprocessor was loaded, attempt to load the newest client-specific preprocessor
    if preprocessor is None:
        patterns = glob.glob(os.path.join(MODEL_DIR, "preprocessor_*.joblib"))
        if patterns:
            latest_pre = max(patterns, key=os.path.getctime)
            try:
                preprocessor = DataPreprocessor.load(latest_pre)
                print(f"Loaded preprocessor from {latest_pre}")
            except Exception as e:
                print(f"Failed to load {latest_pre}: {e}")

    if preprocessor is None:
        print("No preprocessor persisted, fitting a new one on reference data to recover schema...")
        # We generate a small batch of reference data to fit the preprocessor
        ref_df = generate_synthetic_data(n_samples=500)
        preprocessor = DataPreprocessor(
            continuous_cols=CONTINUOUS_COLUMNS,
            categorical_cols=CATEGORICAL_COLUMNS
        )
        preprocessor.fit(ref_df)
    
    print("Applying categorical softmax sampling and inverse transforming data...")
    # Handle continuous and categorical parts separately
    cat_trans = preprocessor.pipeline.named_transformers_.get('cat') if hasattr(preprocessor, 'pipeline') else None
    # (helper function `postprocess_synthetic_array` is defined at module scope and exported)
    
    # Use lower temperature (0.2) for sharper categorical distributions that better match training data
    df_synthetic = postprocess_synthetic_array(preprocessor, synthetic_data, temperature=0.2)
    
    # 5. Distribution Matching: Adjust synthetic data to match real data statistics
    print("Applying distribution matching for continuous features...")
    real_stats = get_real_data_stats()
    for col in CONTINUOUS_COLUMNS:
        if col in df_synthetic.columns and col in real_stats['mean'] and col in real_stats['std']:
            # Get synthetic and real statistics
            synth_mean = df_synthetic[col].mean()
            synth_std = df_synthetic[col].std()
            real_mean = real_stats['mean'][col]
            real_std = real_stats['std'][col]
            
            # Only adjust if there's significant deviation (>10% of real std)
            mean_diff = abs(synth_mean - real_mean)
            std_ratio = synth_std / real_std if real_std > 0 else 1.0
            
            if mean_diff > 0.1 * real_std or std_ratio > 1.3 or std_ratio < 0.7:
                # Z-score normalize then rescale to real distribution
                if synth_std > 0:
                    df_synthetic[col] = (df_synthetic[col] - synth_mean) / synth_std
                    df_synthetic[col] = df_synthetic[col] * real_std + real_mean
    
    # 5b. Apply correlation-aware adjustments
    print("Applying correlation-aware adjustments...")
    df_synthetic = apply_correlation_adjustments(df_synthetic, real_stats)
    
    # 6. Post-processing: Apply clipping based on FEATURE_RANGES from schema
    for col in CONTINUOUS_COLUMNS:
        if col in df_synthetic.columns and col in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[col]
            df_synthetic[col] = df_synthetic[col].clip(min_val, max_val)
            # Round count features to integers
            if col in ['EncounterCount', 'MedicationCount', 'ConditionCount', 'ProcedureCount']:
                df_synthetic[col] = df_synthetic[col].round().astype(int)
            # Round TotalCost to 2 decimal places
            elif col == 'TotalCost':
                df_synthetic[col] = df_synthetic[col].round(2)
    
    # 7. Fix any invalid categorical values by mapping to most common category
    for col in CATEGORICAL_COLUMNS:
        if col in df_synthetic.columns and col in CATEGORIES:
            valid_cats = CATEGORIES[col]
            invalid_mask = ~df_synthetic[col].isin(valid_cats)
            if invalid_mask.any():
                # Replace with first valid category (typically most common)
                df_synthetic.loc[invalid_mask, col] = valid_cats[0]
    
    df_synthetic.to_csv(output_file, index=False)
    print(f"Synthetic data saved to {output_file}")
    return output_file

if __name__ == "__main__":
    generate_synthetic_dataset()
