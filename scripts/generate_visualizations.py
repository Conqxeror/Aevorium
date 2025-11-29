"""
Comprehensive Visualization Generator for Aevorium Synthetic Data
=================================================================
Generates publication-quality visualizations showcasing:
1. Distribution comparisons (real vs synthetic)
2. Correlation heatmaps
3. Quality metrics dashboard
4. Training progress
5. Privacy budget usage
6. Feature-by-feature analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_theme(style="whitegrid", palette="husl")
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not found. Using matplotlib defaults.")

from common.data import generate_synthetic_data
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, CATEGORIES, FEATURE_RANGES
from common.config import MODEL_DIR

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
REAL_COLOR = "#2E86AB"      # Blue for real data
SYNTH_COLOR = "#A23B72"     # Magenta for synthetic
ACCENT_COLOR = "#F18F01"    # Orange for highlights
SUCCESS_COLOR = "#C73E1D"   # Green tones
BG_COLOR = "#F5F5F5"


def load_data():
    """Load real and synthetic datasets."""
    print("Loading datasets...")
    real_df = generate_synthetic_data(1000)
    
    synth_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synthetic_data.csv")
    if not os.path.exists(synth_path):
        print(f"Error: {synth_path} not found. Run generate_samples.py first.")
        sys.exit(1)
    
    synth_df = pd.read_csv(synth_path)
    return real_df, synth_df


def compute_scores(real_df, synth_df):
    """Compute quality scores for each feature."""
    scores = {}
    
    # Continuous features
    for col in CONTINUOUS_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
        r_std, s_std = real_df[col].std(), synth_df[col].std()
        
        col_range = real_df[col].max() - real_df[col].min()
        if col_range > 0:
            mean_error = abs(r_mean - s_mean) / col_range
        else:
            mean_error = 0
        
        if r_std > 0:
            std_error = abs(r_std - s_std) / r_std
        else:
            std_error = 0
        
        scores[col] = 1.0 - min(1.0, 0.6 * mean_error + 0.4 * std_error)
    
    # Categorical features
    for col in CATEGORICAL_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_dist = real_df[col].value_counts(normalize=True)
        s_dist = synth_df[col].value_counts(normalize=True)
        
        all_cats = set(r_dist.index) | set(s_dist.index)
        tvd = 0.5 * sum(abs(r_dist.get(c, 0) - s_dist.get(c, 0)) for c in all_cats)
        scores[col] = 1.0 - min(1.0, tvd)
    
    return scores


def plot_quality_dashboard(real_df, synth_df, scores):
    """Create a comprehensive quality dashboard."""
    print("Generating quality dashboard...")
    
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle("Aevorium Synthetic Data Quality Dashboard", fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Overall Quality Gauge (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    cont_scores = [scores[c] for c in CONTINUOUS_COLUMNS if c in scores]
    cat_scores = [scores[c] for c in CATEGORICAL_COLUMNS if c in scores]
    overall = 0.5 * np.mean(cont_scores) + 0.5 * np.mean(cat_scores) if cont_scores and cat_scores else 0
    
    # Gauge-style visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1
    ax1.fill_between(theta, 0, r, alpha=0.1, color='gray')
    
    # Color zones
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    for i, (start, end) in enumerate([(0, 0.7), (0.7, 0.85), (0.85, 1.0)]):
        mask = (theta >= start * np.pi) & (theta <= end * np.pi)
        ax1.fill_between(theta[mask], 0, r * 0.95, alpha=0.3, color=colors[i])
    
    # Needle
    needle_angle = overall * np.pi
    ax1.plot([0, np.cos(needle_angle) * 0.8], [0, np.sin(needle_angle) * 0.8], 
             color='#333', linewidth=3, solid_capstyle='round')
    ax1.scatter([0], [0], s=100, color='#333', zorder=5)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f"Overall Quality: {overall*100:.1f}%", fontsize=14, fontweight='bold', pad=10)
    
    # 2. Feature Scores Bar Chart (top-middle and top-right)
    ax2 = fig.add_subplot(gs[0, 1:])
    all_features = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
    feature_scores = [scores.get(f, 0) for f in all_features]
    colors = [REAL_COLOR if f in CONTINUOUS_COLUMNS else SYNTH_COLOR for f in all_features]
    
    bars = ax2.barh(range(len(all_features)), [s * 100 for s in feature_scores], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(all_features)))
    ax2.set_yticklabels(all_features, fontsize=9)
    ax2.set_xlabel("Quality Score (%)", fontsize=11)
    ax2.set_xlim(0, 105)
    ax2.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax2.axvline(x=80, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, feature_scores)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{score*100:.1f}%', va='center', fontsize=8)
    
    legend_elements = [mpatches.Patch(facecolor=REAL_COLOR, label='Continuous'),
                       mpatches.Patch(facecolor=SYNTH_COLOR, label='Categorical')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax2.set_title("Feature-Level Quality Scores", fontsize=12, fontweight='bold')
    
    # 3. Distribution Comparison - Age (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(real_df['Age'], bins=25, alpha=0.6, label='Real', color=REAL_COLOR, density=True, edgecolor='white')
    ax3.hist(synth_df['Age'], bins=25, alpha=0.6, label='Synthetic', color=SYNTH_COLOR, density=True, edgecolor='white')
    ax3.set_xlabel("Age", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.set_title(f"Age Distribution (Score: {scores.get('Age', 0)*100:.1f}%)", fontsize=11, fontweight='bold')
    
    # 4. Distribution Comparison - BMI (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(real_df['BMI'], bins=25, alpha=0.6, label='Real', color=REAL_COLOR, density=True, edgecolor='white')
    ax4.hist(synth_df['BMI'], bins=25, alpha=0.6, label='Synthetic', color=SYNTH_COLOR, density=True, edgecolor='white')
    ax4.set_xlabel("BMI", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.set_title(f"BMI Distribution (Score: {scores.get('BMI', 0)*100:.1f}%)", fontsize=11, fontweight='bold')
    
    # 5. Categorical - Gender (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    r_gender = real_df['Gender'].value_counts(normalize=True).sort_index()
    s_gender = synth_df['Gender'].value_counts(normalize=True).sort_index()
    all_genders = sorted(set(r_gender.index) | set(s_gender.index))
    x = np.arange(len(all_genders))
    width = 0.35
    ax5.bar(x - width/2, [r_gender.get(g, 0) for g in all_genders], width, label='Real', color=REAL_COLOR, alpha=0.8)
    ax5.bar(x + width/2, [s_gender.get(g, 0) for g in all_genders], width, label='Synthetic', color=SYNTH_COLOR, alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(all_genders, fontsize=9)
    ax5.set_ylabel("Proportion", fontsize=10)
    ax5.legend(fontsize=9)
    ax5.set_title(f"Gender Distribution (Score: {scores.get('Gender', 0)*100:.1f}%)", fontsize=11, fontweight='bold')
    
    # 6. Categorical - Diagnosis (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    r_diag = real_df['Diagnosis'].value_counts(normalize=True).sort_index()
    s_diag = synth_df['Diagnosis'].value_counts(normalize=True).sort_index()
    all_diag = sorted(set(r_diag.index) | set(s_diag.index))
    x = np.arange(len(all_diag))
    width = 0.35
    ax6.bar(x - width/2, [r_diag.get(d, 0) for d in all_diag], width, label='Real', color=REAL_COLOR, alpha=0.8)
    ax6.bar(x + width/2, [s_diag.get(d, 0) for d in all_diag], width, label='Synthetic', color=SYNTH_COLOR, alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(all_diag, rotation=45, ha='right', fontsize=8)
    ax6.set_ylabel("Proportion", fontsize=10)
    ax6.legend(fontsize=9)
    ax6.set_title(f"Diagnosis Distribution (Score: {scores.get('Diagnosis', 0)*100:.1f}%)", fontsize=11, fontweight='bold')
    
    # 7. Mean Comparison (bottom-middle)
    ax7 = fig.add_subplot(gs[2, 1])
    cont_cols = [c for c in CONTINUOUS_COLUMNS if c in real_df.columns]
    r_means = [real_df[c].mean() for c in cont_cols]
    s_means = [synth_df[c].mean() for c in cont_cols]
    
    # Normalize for comparison
    max_vals = [max(abs(r), abs(s)) for r, s in zip(r_means, s_means)]
    r_norm = [r/m if m > 0 else 0 for r, m in zip(r_means, max_vals)]
    s_norm = [s/m if m > 0 else 0 for s, m in zip(s_means, max_vals)]
    
    x = np.arange(len(cont_cols))
    width = 0.35
    ax7.bar(x - width/2, r_norm, width, label='Real (normalized)', color=REAL_COLOR, alpha=0.8)
    ax7.bar(x + width/2, s_norm, width, label='Synthetic (normalized)', color=SYNTH_COLOR, alpha=0.8)
    ax7.set_xticks(x)
    ax7.set_xticklabels(cont_cols, rotation=45, ha='right', fontsize=8)
    ax7.set_ylabel("Normalized Mean", fontsize=10)
    ax7.legend(fontsize=8, loc='upper right')
    ax7.set_title("Mean Value Comparison", fontsize=11, fontweight='bold')
    
    # 8. Summary Stats (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    📊 SUMMARY STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Overall Quality:     {overall*100:.1f}%
    Continuous Score:    {np.mean(cont_scores)*100:.1f}%
    Categorical Score:   {np.mean(cat_scores)*100:.1f}%
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Total Features:      {len(all_features)}
    Samples Generated:   {len(synth_df)}
    
    Best Feature:        {max(scores, key=scores.get)}
                        ({max(scores.values())*100:.1f}%)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.savefig(os.path.join(OUTPUT_DIR, "quality_dashboard.png"), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'quality_dashboard.png')}")
    plt.close()


def plot_correlation_heatmaps(real_df, synth_df):
    """Generate side-by-side correlation heatmaps."""
    print("Generating correlation heatmaps...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cont_cols = [c for c in CONTINUOUS_COLUMNS if c in real_df.columns]
    
    # Real correlation
    r_corr = real_df[cont_cols].corr()
    if SEABORN_AVAILABLE:
        sns.heatmap(r_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
                    ax=axes[0], square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    else:
        im = axes[0].imshow(r_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_xticks(range(len(cont_cols)))
        axes[0].set_yticks(range(len(cont_cols)))
        axes[0].set_xticklabels(cont_cols, rotation=45, ha='right')
        axes[0].set_yticklabels(cont_cols)
        plt.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].set_title("Real Data Correlations", fontsize=14, fontweight='bold')
    
    # Synthetic correlation
    s_corr = synth_df[cont_cols].corr()
    if SEABORN_AVAILABLE:
        sns.heatmap(s_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=axes[1], square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    else:
        im = axes[1].imshow(s_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(cont_cols)))
        axes[1].set_yticks(range(len(cont_cols)))
        axes[1].set_xticklabels(cont_cols, rotation=45, ha='right')
        axes[1].set_yticklabels(cont_cols)
        plt.colorbar(im, ax=axes[1], shrink=0.8)
    axes[1].set_title("Synthetic Data Correlations", fontsize=14, fontweight='bold')
    
    # Difference
    diff = (r_corr - s_corr).abs()
    if SEABORN_AVAILABLE:
        sns.heatmap(diff, annot=True, fmt='.2f', cmap='Oranges',
                    ax=axes[2], square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    else:
        im = axes[2].imshow(diff, cmap='Oranges', vmin=0, vmax=1)
        axes[2].set_xticks(range(len(cont_cols)))
        axes[2].set_yticks(range(len(cont_cols)))
        axes[2].set_xticklabels(cont_cols, rotation=45, ha='right')
        axes[2].set_yticklabels(cont_cols)
        plt.colorbar(im, ax=axes[2], shrink=0.8)
    axes[2].set_title(f"Absolute Difference (Avg: {diff.mean().mean():.3f})", fontsize=14, fontweight='bold')
    
    plt.suptitle("Correlation Matrix Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmaps.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'correlation_heatmaps.png')}")
    plt.close()


def plot_distribution_grid(real_df, synth_df):
    """Create a grid of all continuous feature distributions."""
    print("Generating distribution grid...")
    
    cont_cols = [c for c in CONTINUOUS_COLUMNS if c in real_df.columns]
    n_cols = 3
    n_rows = (len(cont_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
    
    for i, col in enumerate(cont_cols):
        ax = axes[i]
        
        # KDE plots if seaborn available, else histograms
        if SEABORN_AVAILABLE:
            sns.kdeplot(data=real_df, x=col, ax=ax, color=REAL_COLOR, label='Real', fill=True, alpha=0.3)
            sns.kdeplot(data=synth_df, x=col, ax=ax, color=SYNTH_COLOR, label='Synthetic', fill=True, alpha=0.3)
        else:
            ax.hist(real_df[col], bins=30, alpha=0.5, label='Real', color=REAL_COLOR, density=True)
            ax.hist(synth_df[col], bins=30, alpha=0.5, label='Synthetic', color=SYNTH_COLOR, density=True)
        
        # Add stats
        r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
        ax.axvline(r_mean, color=REAL_COLOR, linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(s_mean, color=SYNTH_COLOR, linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title(f"{col}", fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.legend(fontsize=8)
        
        # Add mean values as text
        ax.text(0.02, 0.98, f"Real μ={r_mean:.1f}\nSynth μ={s_mean:.1f}", 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(len(cont_cols), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Continuous Feature Distributions", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribution_grid.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'distribution_grid.png')}")
    plt.close()


def plot_categorical_comparison(real_df, synth_df):
    """Create detailed categorical feature comparison."""
    print("Generating categorical comparison...")
    
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in real_df.columns]
    n_cols = len(cat_cols)
    
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    
    for i, col in enumerate(cat_cols):
        # Top row: Bar comparison
        ax_bar = axes[0, i] if n_cols > 1 else axes[0]
        r_dist = real_df[col].value_counts(normalize=True).sort_index()
        s_dist = synth_df[col].value_counts(normalize=True).sort_index()
        all_cats = sorted(set(r_dist.index) | set(s_dist.index))
        
        x = np.arange(len(all_cats))
        width = 0.35
        ax_bar.bar(x - width/2, [r_dist.get(c, 0) for c in all_cats], width, 
                   label='Real', color=REAL_COLOR, alpha=0.8)
        ax_bar.bar(x + width/2, [s_dist.get(c, 0) for c in all_cats], width,
                   label='Synthetic', color=SYNTH_COLOR, alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(all_cats, rotation=45, ha='right', fontsize=9)
        ax_bar.set_ylabel("Proportion")
        ax_bar.legend(fontsize=8)
        ax_bar.set_title(f"{col}", fontsize=12, fontweight='bold')
        
        # Bottom row: Difference plot
        ax_diff = axes[1, i] if n_cols > 1 else axes[1]
        diffs = [s_dist.get(c, 0) - r_dist.get(c, 0) for c in all_cats]
        colors = [SUCCESS_COLOR if d >= 0 else ACCENT_COLOR for d in diffs]
        ax_diff.bar(x, diffs, color=colors, alpha=0.8)
        ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_diff.set_xticks(x)
        ax_diff.set_xticklabels(all_cats, rotation=45, ha='right', fontsize=9)
        ax_diff.set_ylabel("Synth - Real")
        ax_diff.set_title("Difference (Synth - Real)", fontsize=10)
    
    plt.suptitle("Categorical Feature Analysis", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "categorical_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'categorical_comparison.png')}")
    plt.close()


def plot_privacy_budget():
    """Visualize privacy budget usage from privacy_budget.json."""
    print("Generating privacy budget visualization...")
    
    privacy_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "privacy_budget.json")
    
    if not os.path.exists(privacy_file):
        print("  Warning: privacy_budget.json not found. Skipping privacy visualization.")
        return
    
    with open(privacy_file, 'r') as f:
        privacy_data = json.load(f)
    
    # Support both 'rounds' and 'round_history' keys
    rounds = privacy_data.get('rounds', privacy_data.get('round_history', []))
    if not rounds:
        print("  Warning: No privacy rounds data found.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    timestamps = [r.get('timestamp', '') for r in rounds]
    epsilons = [r.get('epsilon', 0) for r in rounds]
    cumulative_eps = [r.get('cumulative_epsilon', 0) for r in rounds]
    noise_mults = [r.get('noise_multiplier', 0) for r in rounds]
    losses = [r.get('details', {}).get('final_loss', 0) if isinstance(r.get('details'), dict) else 0 for r in rounds]
    
    round_nums = list(range(1, len(rounds) + 1))
    
    # 1. Epsilon per round
    ax1 = axes[0, 0]
    ax1.bar(round_nums, epsilons, color=REAL_COLOR, alpha=0.8, edgecolor='white')
    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("ε (Privacy Loss)", fontsize=11)
    ax1.set_title("Privacy Budget per Training Step", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Cumulative epsilon
    ax2 = axes[0, 1]
    ax2.plot(round_nums, cumulative_eps, marker='o', color=SYNTH_COLOR, linewidth=2, markersize=4)
    ax2.fill_between(round_nums, cumulative_eps, alpha=0.3, color=SYNTH_COLOR)
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Cumulative ε", fontsize=11)
    ax2.set_title("Cumulative Privacy Budget", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add final value annotation
    if cumulative_eps:
        ax2.annotate(f'Total: {cumulative_eps[-1]:.1f}', 
                     xy=(round_nums[-1], cumulative_eps[-1]),
                     xytext=(round_nums[-1] - len(round_nums)*0.2, cumulative_eps[-1] * 0.85),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='gray'))
    
    # 3. Loss over training
    ax3 = axes[1, 0]
    if any(losses):
        ax3.plot(round_nums, losses, marker='s', color=ACCENT_COLOR, linewidth=2, markersize=4)
        ax3.set_xlabel("Training Step", fontsize=11)
        ax3.set_ylabel("Final Loss", fontsize=11)
        ax3.set_title("Training Loss Progress", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(round_nums, losses, 1)
        p = np.poly1d(z)
        ax3.plot(round_nums, p(round_nums), linestyle='--', color='gray', alpha=0.7, label='Trend')
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Training Loss Progress", fontsize=12, fontweight='bold')
    
    # 4. Privacy-Utility Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_eps = privacy_data.get('cumulative_epsilon', cumulative_eps[-1] if cumulative_eps else 0)
    total_steps = len(rounds)
    avg_eps = total_eps / max(total_steps, 1)
    delta = privacy_data.get('delta', 1e-5)
    
    # Calculate loss improvement
    if losses and len(losses) > 1:
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
    else:
        improvement = 0
    
    initial_loss_str = f"{losses[0]:.4f}" if losses else 'N/A'
    final_loss_str = f"{losses[-1]:.4f}" if losses else 'N/A'
    
    text = f"""
    PRIVACY & TRAINING SUMMARY
    ════════════════════════════════════════
    
    Privacy Metrics:
    ────────────────────────────────────────
    • Total Training Steps:    {total_steps}
    • Cumulative ε:            {total_eps:.2f}
    • Average ε per Step:      {avg_eps:.2f}
    • Delta (δ):               {delta:.0e}
    • Noise Multiplier:        {noise_mults[0] if noise_mults else 'N/A'}
    
    Training Metrics:
    ────────────────────────────────────────
    • Initial Loss:            {initial_loss_str}
    • Final Loss:              {final_loss_str}
    • Loss Improvement:        {improvement:.1f}%
    
    Privacy Guarantee:
    ────────────────────────────────────────
    ({total_eps:.1f}, {delta:.0e})-Differential Privacy
    
    """
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6'))
    
    plt.suptitle("Differential Privacy & Training Analysis", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "privacy_budget.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'privacy_budget.png')}")
    plt.close()


def plot_training_progress():
    """Visualize training progress from audit log."""
    print("Generating training progress visualization...")
    
    audit_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audit_log.json")
    
    if not os.path.exists(audit_file):
        print("  Warning: audit_log.json not found. Skipping training progress.")
        return
    
    events = []
    with open(audit_file, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                pass
    
    # Filter for training events
    training_events = [e for e in events if 'round' in str(e.get('details', {})).lower() 
                       or 'aggregat' in str(e.get('event', '')).lower()]
    
    if not training_events:
        print("  Warning: No training events found in audit log.")
        return
    
    # Extract round info
    rounds_data = []
    for e in training_events:
        details = e.get('details', {})
        if isinstance(details, dict) and 'round' in details:
            rounds_data.append({
                'round': details.get('round', 0),
                'timestamp': e.get('timestamp', ''),
                'event': e.get('event', '')
            })
    
    if not rounds_data:
        # Try to get from other patterns
        print("  Warning: Could not extract round data. Creating basic visualization.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    round_nums = list(range(1, len(rounds_data) + 1))
    
    ax.step(round_nums, round_nums, where='mid', linewidth=2, color=REAL_COLOR, marker='o')
    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel("Completed Rounds", fontsize=12)
    ax.set_title("Federated Learning Training Progress", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_progress.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'training_progress.png')}")
    plt.close()


def plot_feature_radar(scores):
    """Create a radar chart of feature quality scores."""
    print("Generating feature radar chart...")
    
    features = list(scores.keys())
    values = [scores[f] * 100 for f in features]
    
    # Number of variables
    N = len(features)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # Complete the loop
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, values, 'o-', linewidth=2, color=REAL_COLOR)
    ax.fill(angles, values, alpha=0.25, color=REAL_COLOR)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_ylim(0, 100)
    
    # Add reference circles
    for val in [80, 90]:
        circle = plt.Circle((0, 0), val, transform=ax.transData._b, 
                            fill=False, color='green' if val == 90 else 'orange', 
                            linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    plt.title("Feature Quality Radar", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_radar.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'feature_radar.png')}")
    plt.close()


def generate_html_report(real_df, synth_df, scores):
    """Generate an HTML report with embedded images."""
    print("Generating HTML report...")
    
    cont_scores = [scores[c] for c in CONTINUOUS_COLUMNS if c in scores]
    cat_scores = [scores[c] for c in CATEGORICAL_COLUMNS if c in scores]
    overall = 0.5 * np.mean(cont_scores) + 0.5 * np.mean(cat_scores) if cont_scores and cat_scores else 0
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aevorium Synthetic Data Quality Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #A23B72;
            margin-top: 40px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: center;
        }}
        .summary-box h2 {{
            color: white;
            margin-top: 0;
        }}
        .score {{
            font-size: 72px;
            font-weight: bold;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #2E86AB;
        }}
        .metric-label {{
            color: #666;
            margin-top: 10px;
        }}
        img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2E86AB;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <h1>🧬 Aevorium Synthetic Data Quality Report</h1>
    
    <div class="summary-box">
        <h2>Overall Quality Score</h2>
        <div class="score">{overall*100:.1f}%</div>
        <p>Generated {len(synth_df)} synthetic healthcare records</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{np.mean(cont_scores)*100:.1f}%</div>
            <div class="metric-label">Continuous Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{np.mean(cat_scores)*100:.1f}%</div>
            <div class="metric-label">Categorical Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(scores)}</div>
            <div class="metric-label">Total Features</div>
        </div>
    </div>
    
    <h2>📊 Quality Dashboard</h2>
    <img src="quality_dashboard.png" alt="Quality Dashboard">
    
    <h2>📈 Feature Distributions</h2>
    <img src="distribution_grid.png" alt="Distribution Grid">
    
    <h2>🔗 Correlation Analysis</h2>
    <img src="correlation_heatmaps.png" alt="Correlation Heatmaps">
    
    <h2>📋 Categorical Features</h2>
    <img src="categorical_comparison.png" alt="Categorical Comparison">
    
    <h2>🎯 Feature Radar</h2>
    <img src="feature_radar.png" alt="Feature Radar">
    
    <h2>🔒 Privacy Budget</h2>
    <img src="privacy_budget.png" alt="Privacy Budget">
    
    <h2>📝 Detailed Scores</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Type</th>
            <th>Quality Score</th>
        </tr>
        {"".join(f'<tr><td>{f}</td><td>{"Continuous" if f in CONTINUOUS_COLUMNS else "Categorical"}</td><td class="{"good" if scores[f] >= 0.9 else "warning"}">{scores[f]*100:.1f}%</td></tr>' for f in scores)}
    </table>
    
    <div class="footer">
        <p>Generated by Aevorium Federated Learning Platform</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
    
    report_path = os.path.join(OUTPUT_DIR, "quality_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved: {report_path}")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("AEVORIUM VISUALIZATION GENERATOR")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load data
    real_df, synth_df = load_data()
    
    # Compute scores
    scores = compute_scores(real_df, synth_df)
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    print("-" * 40)
    
    plot_quality_dashboard(real_df, synth_df, scores)
    plot_correlation_heatmaps(real_df, synth_df)
    plot_distribution_grid(real_df, synth_df)
    plot_categorical_comparison(real_df, synth_df)
    plot_feature_radar(scores)
    plot_privacy_budget()
    plot_training_progress()
    generate_html_report(real_df, synth_df, scores)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  • {f}")
    
    print(f"\n💡 Open {os.path.join(OUTPUT_DIR, 'quality_report.html')} in a browser for the full report!")


if __name__ == "__main__":
    main()
