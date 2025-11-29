import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib not found. Plots will be skipped.")
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.data import generate_synthetic_data

from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS

def compare_statistics(real_df, synth_df):
    print("\n--- Statistical Comparison (Continuous) ---")
    print(f"{'Feature':<15} | {'Real Mean':<10} | {'Synth Mean':<10} | {'Real Std':<10} | {'Synth Std':<10}")
    print("-" * 70)
    
    for col in CONTINUOUS_COLUMNS:
        if col not in real_df.columns: continue
        r_mean = real_df[col].mean()
        s_mean = synth_df[col].mean()
        r_std = real_df[col].std()
        s_std = synth_df[col].std()
        print(f"{col:<15} | {r_mean:<10.2f} | {s_mean:<10.2f} | {r_std:<10.2f} | {s_std:<10.2f}")

    print("\n--- Categorical Distribution Comparison ---")
    for col in CATEGORICAL_COLUMNS:
        if col not in real_df.columns: continue
        print(f"\nFeature: {col}")
        r_counts = real_df[col].value_counts(normalize=True).sort_index()
        s_counts = synth_df[col].value_counts(normalize=True).sort_index()
        
        df_compare = pd.DataFrame({'Real': r_counts, 'Synth': s_counts}).fillna(0)
        print(df_compare.round(3))

def compare_correlations(real_df, synth_df):
    print("\n--- Correlation Matrix Comparison (Continuous) ---")
    # Only use continuous columns
    r_corr = real_df[CONTINUOUS_COLUMNS].corr()
    s_corr = synth_df[CONTINUOUS_COLUMNS].corr()
    
    print("Real Data Correlation:")
    print(r_corr.round(2))
    print("\nSynthetic Data Correlation:")
    print(s_corr.round(2))
    
    diff = (r_corr - s_corr).abs().mean().mean()
    print(f"\nAverage Absolute Correlation Difference: {diff:.4f}")

def plot_histograms(real_df, synth_df):
    # Plot Continuous
    n_cont = len(CONTINUOUS_COLUMNS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # Assuming 4 continuous vars
    axes = axes.flatten()
    
    for i, col in enumerate(CONTINUOUS_COLUMNS):
        if i >= len(axes): break
        ax = axes[i]
        ax.hist(real_df[col], bins=30, alpha=0.5, label='Real', density=True)
        ax.hist(synth_df[col], bins=30, alpha=0.5, label='Synthetic', density=True)
        ax.set_title(col)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("validation_plots_continuous.png")
    print("\nContinuous plots saved to validation_plots_continuous.png")
    
    # Plot Categorical
    n_cat = len(CATEGORICAL_COLUMNS)
    fig2, axes2 = plt.subplots(1, n_cat, figsize=(12, 5))
    if n_cat == 1: axes2 = [axes2]
    
    for i, col in enumerate(CATEGORICAL_COLUMNS):
        ax = axes2[i]
        r_counts = real_df[col].value_counts(normalize=True).sort_index()
        s_counts = synth_df[col].value_counts(normalize=True).sort_index()
        
        # Align indices
        all_cats = sorted(list(set(r_counts.index) | set(s_counts.index)))
        r_vals = [r_counts.get(c, 0) for c in all_cats]
        s_vals = [s_counts.get(c, 0) for c in all_cats]
        
        x = np.arange(len(all_cats))
        width = 0.35
        
        ax.bar(x - width/2, r_vals, width, label='Real')
        ax.bar(x + width/2, s_vals, width, label='Synthetic')
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, rotation=45)
        ax.set_title(col)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig("validation_plots_categorical.png")
    print("Categorical plots saved to validation_plots_categorical.png")

def compute_quality_score(real_df, synth_df):
    """
    Compute an overall quality score based on:
    1. Mean absolute error for continuous columns (normalized)
    2. Distribution difference for categorical columns
    3. Std deviation difference for continuous columns
    """
    scores = []
    
    # Continuous: mean & std matching
    for col in CONTINUOUS_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
        r_std, s_std = real_df[col].std(), synth_df[col].std()
        
        # Relative error for mean (normalized by range to avoid div-by-zero for small means)
        col_range = real_df[col].max() - real_df[col].min()
        if col_range > 0:
            mean_error = abs(r_mean - s_mean) / col_range
        else:
            mean_error = 0 if abs(r_mean - s_mean) < 0.001 else 1
        
        # Relative error for std
        if r_std > 0:
            std_error = abs(r_std - s_std) / r_std
        else:
            std_error = 0 if s_std < 0.001 else 1
        
        # Combine (mean is more important than std)
        col_score = 1.0 - min(1.0, 0.6 * mean_error + 0.4 * std_error)
        scores.append(('cont_' + col, col_score))
    
    # Categorical: Total Variation Distance (TVD)
    for col in CATEGORICAL_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_dist = real_df[col].value_counts(normalize=True)
        s_dist = synth_df[col].value_counts(normalize=True)
        
        # Align distributions
        all_cats = set(r_dist.index) | set(s_dist.index)
        tvd = 0.5 * sum(abs(r_dist.get(c, 0) - s_dist.get(c, 0)) for c in all_cats)
        col_score = 1.0 - min(1.0, tvd)
        scores.append(('cat_' + col, col_score))
    
    # Compute weighted average (categoricals weighted slightly higher since they're easier)
    cont_scores = [s for name, s in scores if name.startswith('cont_')]
    cat_scores = [s for name, s in scores if name.startswith('cat_')]
    
    cont_avg = sum(cont_scores) / len(cont_scores) if cont_scores else 0
    cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
    
    # Balanced weighting
    overall = 0.5 * cont_avg + 0.5 * cat_avg
    
    print("\n" + "=" * 70)
    print("QUALITY SCORE SUMMARY")
    print("=" * 70)
    print(f"Continuous Features Score: {cont_avg * 100:.1f}%")
    for name, s in scores:
        if name.startswith('cont_'):
            print(f"  {name[5:]}: {s * 100:.1f}%")
    print(f"\nCategorical Features Score: {cat_avg * 100:.1f}%")
    for name, s in scores:
        if name.startswith('cat_'):
            print(f"  {name[4:]}: {s * 100:.1f}%")
    print(f"\nOVERALL QUALITY SCORE: {overall * 100:.1f}%")
    print("=" * 70)
    
    return overall

def main():
    # 1. Load Data
    print("Generating Ground Truth Data...")
    real_df = generate_synthetic_data(1000)
    
    if not os.path.exists("synthetic_data.csv"):
        print("Error: synthetic_data.csv not found. Run generate_samples.py first.")
        return
        
    print("Loading Synthetic Data...")
    synth_df = pd.read_csv("synthetic_data.csv")
    
    # 2. Compare
    compare_statistics(real_df, synth_df)
    compare_correlations(real_df, synth_df)
    
    # 3. Compute Quality Score
    compute_quality_score(real_df, synth_df)
    
    # 4. Plot
    if PLOT_AVAILABLE:
        try:
            plot_histograms(real_df, synth_df)
        except Exception as e:
            print(f"Could not generate plots: {e}")

if __name__ == "__main__":
    main()
