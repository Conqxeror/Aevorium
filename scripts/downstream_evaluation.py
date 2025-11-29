"""
Downstream Task Evaluation for Synthetic Data Utility

This script measures the practical utility of synthetic data by:
1. Train on Real, Test on Real (TRTR) - Baseline accuracy
2. Train on Synthetic, Test on Real (TSTR) - Synthetic data utility
3. Train on Real, Test on Synthetic (TRTS) - Distribution match check

A high TSTR accuracy close to TRTR indicates the synthetic data preserves
the statistical relationships needed for downstream ML tasks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from common.data import load_real_healthcare_data
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, CATEGORIES


def prepare_features(df, target_col='Diagnosis', exclude_cols=None, categories_map=None):
    """
    Prepare features for ML model training.
    Encodes categorical features and scales continuous features.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        exclude_cols: Columns to exclude
        categories_map: Dict of column -> list of categories for consistent one-hot encoding
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Make a copy
    df = df.copy()
    
    # Separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    
    y = df[target_col]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Features: continuous + one-hot encoded categorical (excluding target)
    feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
    
    # Separate continuous and categorical
    cont_cols = [c for c in CONTINUOUS_COLUMNS if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols and c != target_col]
    
    # Build feature matrix
    X_parts = []
    
    # Continuous features
    if cont_cols:
        X_cont = df[cont_cols].values.astype(float)
        X_parts.append(X_cont)
    
    # Categorical features (one-hot with consistent categories)
    if cat_cols:
        if categories_map:
            # Use predefined categories for consistent encoding
            cat_dfs = []
            for col in cat_cols:
                if col in categories_map:
                    # Create dummy columns for all possible categories
                    for cat in categories_map[col]:
                        cat_dfs.append((df[col] == cat).astype(float).values.reshape(-1, 1))
                else:
                    # Fallback to regular dummies
                    dummies = pd.get_dummies(df[col], prefix=col)
                    cat_dfs.append(dummies.values)
            X_cat = np.hstack(cat_dfs)
        else:
            X_cat = pd.get_dummies(df[cat_cols], drop_first=False).values.astype(float)
        X_parts.append(X_cat)
    
    X = np.hstack(X_parts) if X_parts else np.array([])
    
    return X, y_encoded, le


def evaluate_downstream_tasks(real_df, synth_df, target_col='Diagnosis'):
    """
    Comprehensive downstream task evaluation.
    
    Metrics:
    - TRTR: Train on Real, Test on Real (baseline)
    - TSTR: Train on Synthetic, Test on Real (utility metric)
    - TRTS: Train on Real, Test on Synthetic (sanity check)
    """
    print("=" * 70)
    print("DOWNSTREAM TASK EVALUATION")
    print("=" * 70)
    print(f"\nTarget Variable: {target_col}")
    print(f"Real Data Size: {len(real_df)}")
    print(f"Synthetic Data Size: {len(synth_df)}")
    
    # Prepare features with consistent categories
    print("\nPreparing features...")
    
    # Ensure both datasets have same columns
    common_cols = list(set(real_df.columns) & set(synth_df.columns))
    real_df = real_df[common_cols]
    synth_df = synth_df[common_cols]
    
    # Build categories map for consistent encoding (excluding target)
    cat_map = {k: v for k, v in CATEGORIES.items() if k != target_col}
    
    X_real, y_real, le = prepare_features(real_df, target_col, categories_map=cat_map)
    X_synth, y_synth, _ = prepare_features(synth_df, target_col, categories_map=cat_map)
    
    # Handle NaN/Inf values
    X_real = np.nan_to_num(X_real, nan=0, posinf=0, neginf=0)
    X_synth = np.nan_to_num(X_synth, nan=0, posinf=0, neginf=0)
    
    print(f"Feature dimensions: {X_real.shape[1]}")
    print(f"Classes: {le.classes_}")
    
    # Split real data for testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
    )
    
    # Scale features
    scaler = StandardScaler()
    X_real_train_scaled = scaler.fit_transform(X_real_train)
    X_real_test_scaled = scaler.transform(X_real_test)
    X_synth_scaled = scaler.transform(X_synth)
    
    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print('=' * 50)
        
        results[model_name] = {}
        
        # =========================================
        # TRTR: Train on Real, Test on Real
        # =========================================
        print("\n[TRTR] Train on Real, Test on Real (Baseline)")
        model_trtr = models[model_name].__class__(**model.get_params())
        model_trtr.fit(X_real_train_scaled, y_real_train)
        
        y_pred_trtr = model_trtr.predict(X_real_test_scaled)
        acc_trtr = accuracy_score(y_real_test, y_pred_trtr)
        f1_trtr = f1_score(y_real_test, y_pred_trtr, average='weighted')
        
        print(f"  Accuracy: {acc_trtr:.4f}")
        print(f"  F1 Score: {f1_trtr:.4f}")
        
        results[model_name]['TRTR'] = {'accuracy': acc_trtr, 'f1': f1_trtr}
        
        # =========================================
        # TSTR: Train on Synthetic, Test on Real
        # =========================================
        print("\n[TSTR] Train on Synthetic, Test on Real (Utility)")
        model_tstr = models[model_name].__class__(**model.get_params())
        model_tstr.fit(X_synth_scaled, y_synth)
        
        y_pred_tstr = model_tstr.predict(X_real_test_scaled)
        acc_tstr = accuracy_score(y_real_test, y_pred_tstr)
        f1_tstr = f1_score(y_real_test, y_pred_tstr, average='weighted')
        
        print(f"  Accuracy: {acc_tstr:.4f}")
        print(f"  F1 Score: {f1_tstr:.4f}")
        
        results[model_name]['TSTR'] = {'accuracy': acc_tstr, 'f1': f1_tstr}
        
        # Utility ratio
        utility_ratio = acc_tstr / acc_trtr if acc_trtr > 0 else 0
        print(f"  Utility Ratio (TSTR/TRTR): {utility_ratio:.4f}")
        results[model_name]['utility_ratio'] = utility_ratio
        
        # =========================================
        # TRTS: Train on Real, Test on Synthetic
        # =========================================
        print("\n[TRTS] Train on Real, Test on Synthetic (Sanity)")
        y_pred_trts = model_trtr.predict(X_synth_scaled)
        acc_trts = accuracy_score(y_synth, y_pred_trts)
        f1_trts = f1_score(y_synth, y_pred_trts, average='weighted')
        
        print(f"  Accuracy: {acc_trts:.4f}")
        print(f"  F1 Score: {f1_trts:.4f}")
        
        results[model_name]['TRTS'] = {'accuracy': acc_trts, 'f1': f1_trts}
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'TRTR Acc': f"{metrics['TRTR']['accuracy']:.4f}",
            'TSTR Acc': f"{metrics['TSTR']['accuracy']:.4f}",
            'TRTS Acc': f"{metrics['TRTS']['accuracy']:.4f}",
            'Utility': f"{metrics['utility_ratio']:.2%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Overall utility score
    avg_utility = np.mean([m['utility_ratio'] for m in results.values()])
    print(f"\nOverall Synthetic Data Utility: {avg_utility:.2%}")
    
    if avg_utility >= 0.9:
        print("✓ EXCELLENT: Synthetic data preserves >90% of ML utility")
    elif avg_utility >= 0.7:
        print("◐ GOOD: Synthetic data preserves 70-90% of ML utility")
    elif avg_utility >= 0.5:
        print("△ MODERATE: Synthetic data preserves 50-70% of ML utility")
    else:
        print("✗ POOR: Synthetic data preserves <50% of ML utility")
    
    return results


def evaluate_multiple_targets(real_df, synth_df):
    """
    Evaluate utility across multiple prediction targets.
    """
    targets = ['Diagnosis', 'RiskLevel', 'HasAllergies']
    
    all_results = {}
    for target in targets:
        if target in real_df.columns and target in synth_df.columns:
            print(f"\n\n{'#' * 70}")
            print(f"# TARGET: {target}")
            print('#' * 70)
            all_results[target] = evaluate_downstream_tasks(real_df, synth_df, target)
    
    return all_results


if __name__ == "__main__":
    print("Loading Real Data...")
    real_df = load_real_healthcare_data()
    
    print("\nLoading Synthetic Data...")
    synth_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'synthetic_data.csv')
    
    if not os.path.exists(synth_path):
        print(f"Error: Synthetic data not found at {synth_path}")
        print("Please run generate_samples.py first.")
        sys.exit(1)
    
    synth_df = pd.read_csv(synth_path)
    
    print(f"\nReal data shape: {real_df.shape}")
    print(f"Synthetic data shape: {synth_df.shape}")
    
    # Run evaluation for primary target
    results = evaluate_downstream_tasks(real_df, synth_df, target_col='Diagnosis')
    
    # Optional: evaluate for RiskLevel if present
    if 'RiskLevel' in real_df.columns and 'RiskLevel' in synth_df.columns:
        print("\n\n" + "#" * 70)
        print("# ADDITIONAL TARGET: RiskLevel")
        print("#" * 70)
        results_risk = evaluate_downstream_tasks(real_df, synth_df, target_col='RiskLevel')
