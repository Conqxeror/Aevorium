"""
Aevorium Monitoring Dashboard

A Streamlit-based real-time dashboard for monitoring:
- Federated Learning training progress
- Synthetic data quality metrics
- Privacy budget consumption
- System health
- Data generation controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import time
from datetime import datetime
import requests
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.governance import get_logs, get_privacy_tracker
from common.data import load_real_healthcare_data
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, CATEGORIES, FEATURE_RANGES

# API endpoint configuration
API_BASE_URL = os.getenv("AEVORIUM_API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Aevorium Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-danger { color: #dc3545; font-weight: bold; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .api-online { background-color: #d4edda; color: #155724; }
    .api-offline { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is online"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def api_generate_samples(n_samples: int, output_file: str = "synthetic_data.csv"):
    """Call API to generate synthetic samples"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={"n_samples": n_samples, "output_file": output_file},
            timeout=120
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_get_privacy_budget():
    """Get privacy budget from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/privacy-budget", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def api_set_privacy_limit(total_budget: float):
    """Set privacy budget limit via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/privacy-budget/set-limit",
            json={"total_budget": total_budget},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def api_reset_privacy():
    """Reset privacy budget via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/privacy-budget/reset", timeout=5)
        return response.status_code == 200
    except:
        return False


def load_synthetic_data():
    """Load synthetic data if available"""
    synth_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'synthetic_data.csv')
    if os.path.exists(synth_path):
        return pd.read_csv(synth_path)
    return None


def get_model_files():
    """Get list of trained model checkpoints"""
    model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = []
    for f in os.listdir(model_dir):
        if f.startswith('global_model_round_') and f.endswith('.npz'):
            round_num = int(f.replace('global_model_round_', '').replace('.npz', ''))
            stat = os.stat(os.path.join(model_dir, f))
            files.append({
                'file': f,
                'round': round_num,
                'size_kb': stat.st_size / 1024,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
    return sorted(files, key=lambda x: x['round'])


def compute_quality_score(real_df, synth_df):
    """Compute quality score similar to validate_data.py"""
    scores = []
    
    # Continuous: mean & std matching
    for col in CONTINUOUS_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_mean, s_mean = real_df[col].mean(), synth_df[col].mean()
        r_std, s_std = real_df[col].std(), synth_df[col].std()
        
        col_range = real_df[col].max() - real_df[col].min()
        if col_range > 0:
            mean_error = abs(r_mean - s_mean) / col_range
        else:
            mean_error = 0 if abs(r_mean - s_mean) < 0.001 else 1
        
        if r_std > 0:
            std_error = abs(r_std - s_std) / r_std
        else:
            std_error = 0 if s_std < 0.001 else 1
        
        col_score = 1.0 - min(1.0, 0.6 * mean_error + 0.4 * std_error)
        scores.append(('cont', col, col_score))
    
    # Categorical: TVD
    for col in CATEGORICAL_COLUMNS:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r_dist = real_df[col].value_counts(normalize=True)
        s_dist = synth_df[col].value_counts(normalize=True)
        
        all_cats = set(r_dist.index) | set(s_dist.index)
        tvd = 0.5 * sum(abs(r_dist.get(c, 0) - s_dist.get(c, 0)) for c in all_cats)
        col_score = 1.0 - min(1.0, tvd)
        scores.append(('cat', col, col_score))
    
    cont_scores = [s for t, _, s in scores if t == 'cont']
    cat_scores = [s for t, _, s in scores if t == 'cat']
    
    cont_avg = sum(cont_scores) / len(cont_scores) if cont_scores else 0
    cat_avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
    
    overall = 0.5 * cont_avg + 0.5 * cat_avg
    return overall, cont_avg, cat_avg, scores


def main():
    # Header
    st.markdown('<div class="main-header">🔬 Aevorium Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Federated Learning Platform for Synthetic Healthcare Data Generation**")
    
    # API Status indicator
    api_online = check_api_health()
    if api_online:
        st.markdown('<span class="api-status api-online">✓ API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="api-status api-offline">✗ API Offline</span>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔬 Aevorium")
    st.sidebar.markdown("---")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(0.1)  # Small delay to prevent immediate refresh
        st.sidebar.info("Refreshing every 30 seconds...")
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigation", [
        "📊 Overview",
        "🚀 Generate Data",
        "🔐 Privacy Budget",
        "📈 Data Quality",
        "📋 Audit Log",
        "⚙️ Training History"
    ])
    
    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 2.0.0")
    st.sidebar.markdown(f"**API:** `{API_BASE_URL}`")
    
    if page == "📊 Overview":
        show_overview()
    elif page == "🚀 Generate Data":
        show_generation()
    elif page == "🔐 Privacy Budget":
        show_privacy_budget()
    elif page == "📈 Data Quality":
        show_data_quality()
    elif page == "📋 Audit Log":
        show_audit_log()
    elif page == "⚙️ Training History":
        show_training_history()
    
    # Auto-refresh trigger
    if auto_refresh:
        time.sleep(30)
        st.rerun()


def show_overview():
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Model info
    model_files = get_model_files()
    synth_data = load_synthetic_data()
    privacy_tracker = get_privacy_tracker()
    logs = get_logs()
    
    with col1:
        st.metric("Training Rounds", len(model_files), 
                  delta="Latest: Round " + str(model_files[-1]['round']) if model_files else "No training")
    
    with col2:
        if synth_data is not None:
            st.metric("Synthetic Samples", f"{len(synth_data):,}")
        else:
            st.metric("Synthetic Samples", "N/A")
    
    with col3:
        summary = privacy_tracker.get_summary()
        st.metric("Privacy ε Spent", f"{summary['cumulative_epsilon']:.2f}",
                  delta=f"{summary['avg_epsilon_per_round']:.1f} per round" if summary['num_rounds'] > 0 else None)
    
    with col4:
        st.metric("Audit Events", len(logs))
    
    # Quality score if data available
    if synth_data is not None:
        try:
            real_data = load_real_healthcare_data()
            if real_data is not None:
                overall, cont, cat, _ = compute_quality_score(real_data, synth_data)
                st.markdown("---")
                st.subheader("📊 Quality Score")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Quality", f"{overall*100:.1f}%")
                with col2:
                    st.metric("Continuous Features", f"{cont*100:.1f}%")
                with col3:
                    st.metric("Categorical Features", f"{cat*100:.1f}%")
        except Exception:
            pass
    
    st.markdown("---")
    
    # Schema info with collapsible sections
    st.subheader("📋 Data Schema")
    
    with st.expander("Continuous Features (9)", expanded=False):
        schema_data = []
        for col in CONTINUOUS_COLUMNS:
            range_info = FEATURE_RANGES.get(col, (None, None))
            schema_data.append({
                'Feature': col,
                'Min': range_info[0],
                'Max': range_info[1]
            })
        st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)
    
    with st.expander("Categorical Features (5)", expanded=False):
        for col in CATEGORICAL_COLUMNS:
            cats = CATEGORIES.get(col, [])
            st.write(f"**{col}:** {', '.join(cats)}")
    
    # Recent activity
    st.markdown("---")
    st.subheader("📜 Recent Activity")
    if logs:
        recent_logs = logs[-5:][::-1]
        for log in recent_logs:
            ts = log.get('timestamp', '')[:19]
            event = log.get('event_type', 'Unknown')
            if event.startswith('privacy'):
                icon = "🔐"
            elif event == 'MODEL_SAVED':
                icon = "💾"
            elif event == 'DATA_GENERATION':
                icon = "📊"
            else:
                icon = "🔹"
            st.write(f"{icon} [{ts}] **{event}**")
    else:
        st.info("No activity logged yet. Start training to see events here.")


def show_generation():
    st.header("🚀 Generate Synthetic Data")
    
    # Check for trained model
    model_files = get_model_files()
    
    if not model_files:
        st.warning("⚠️ No trained model found. Please run training first.")
        st.code("""
# Start the server
python server/server.py

# Start 2+ clients (in separate terminals)
python node/client.py
        """)
        return
    
    st.success(f"✓ Model available: Round {model_files[-1]['round']}")
    
    # Generation options
    st.subheader("Generation Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.number_input(
            "Number of samples",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
            help="Number of synthetic patient records to generate"
        )
    
    with col2:
        output_file = st.text_input(
            "Output filename",
            value="synthetic_data.csv",
            help="Filename for the generated data"
        )
    
    # Generate button
    st.markdown("---")
    
    api_online = check_api_health()
    
    if api_online:
        st.info("🌐 API is online. Generation will use the API endpoint.")
        
        if st.button("🚀 Generate via API", type="primary", use_container_width=True):
            with st.spinner(f"Generating {n_samples:,} samples..."):
                result = api_generate_samples(n_samples, output_file)
                if result:
                    st.success(f"✓ Generated {n_samples:,} samples to `{result.get('path', output_file)}`")
                    st.balloons()
                else:
                    st.error("Failed to generate samples via API")
    else:
        st.warning("⚠️ API is offline. Using local generation.")
        
        if st.button("🚀 Generate Locally", type="primary", use_container_width=True):
            with st.spinner(f"Generating {n_samples:,} samples..."):
                try:
                    # Import and run local generation
                    from generate_samples import generate_synthetic_dataset
                    result_path = generate_synthetic_dataset(n_samples, output_file)
                    st.success(f"✓ Generated {n_samples:,} samples to `{result_path}`")
                    st.balloons()
                except Exception as e:
                    st.error(f"Generation failed: {e}")
    
    # Show existing synthetic data
    st.markdown("---")
    st.subheader("Current Synthetic Data")
    
    synth_data = load_synthetic_data()
    
    if synth_data is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(synth_data):,}")
        with col2:
            st.metric("Features", len(synth_data.columns))
        with col3:
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'synthetic_data.csv')
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                st.metric("File Size", f"{size_mb:.2f} MB")
        
        # Preview data
        with st.expander("Preview Data", expanded=True):
            st.dataframe(synth_data.head(20), use_container_width=True)
        
        # Download button
        csv = synth_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=output_file,
            mime="text/csv",
            use_container_width=True
        )
        
        # Quick stats
        with st.expander("Quick Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Continuous Features**")
                st.dataframe(synth_data[CONTINUOUS_COLUMNS].describe().round(2))
            with col2:
                st.write("**Categorical Distributions**")
                for cat in CATEGORICAL_COLUMNS:
                    if cat in synth_data.columns:
                        st.write(f"*{cat}:*")
                        st.write(synth_data[cat].value_counts(normalize=True).round(3).to_dict())
    else:
        st.info("No synthetic data generated yet. Click 'Generate' above to create data.")


def show_privacy_budget():
    st.header("🔐 Privacy Budget Tracking")
    
    # Try API first, fallback to local
    api_budget = api_get_privacy_budget() if check_api_health() else None
    
    if api_budget:
        summary = api_budget
    else:
        tracker = get_privacy_tracker()
        summary = tracker.get_summary()
    
    # Budget overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cumulative ε", f"{summary['cumulative_epsilon']:.2f}")
    
    with col2:
        if summary['total_budget']:
            remaining = summary['budget_remaining']
            st.metric("Budget Remaining", f"{remaining:.2f}")
        else:
            st.metric("Budget Remaining", "∞ (No limit)")
    
    with col3:
        if summary['total_budget']:
            pct = summary['budget_exhausted_pct']
            if pct < 50:
                st.metric("Budget Used", f"{pct:.1f}%", delta="Safe", delta_color="normal")
            elif pct < 80:
                st.metric("Budget Used", f"{pct:.1f}%", delta="Warning", delta_color="off")
            else:
                st.metric("Budget Used", f"{pct:.1f}%", delta="Critical", delta_color="inverse")
        else:
            st.metric("Budget Used", "N/A")
    
    # Budget gauge
    if summary['total_budget']:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=summary['cumulative_epsilon'],
            delta={'reference': summary['total_budget'] * 0.5},
            gauge={
                'axis': {'range': [0, summary['total_budget']]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, summary['total_budget'] * 0.5], 'color': "lightgreen"},
                    {'range': [summary['total_budget'] * 0.5, summary['total_budget'] * 0.8], 'color': "yellow"},
                    {'range': [summary['total_budget'] * 0.8, summary['total_budget']], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': summary['total_budget']
                }
            },
            title={'text': "Privacy Budget (ε)"}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Round history
    st.subheader("Training Round History")
    if summary['round_history']:
        history_df = pd.DataFrame(summary['round_history'])
        if 'timestamp' in history_df.columns:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Epsilon over rounds
        fig = px.line(history_df, x='round_num', y='cumulative_epsilon',
                      markers=True, title="Cumulative Privacy Expenditure")
        fig.update_layout(xaxis_title="Round", yaxis_title="Cumulative ε")
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(history_df[['round_num', 'epsilon', 'cumulative_epsilon', 'noise_multiplier', 'epochs']])
    else:
        st.info("No training rounds recorded yet.")
    
    # Set budget limit
    st.subheader("⚙️ Configure Budget")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Set Privacy Budget Limit"):
            new_budget = st.number_input("Total Privacy Budget (ε)", min_value=0.0, value=100.0, step=10.0)
            if st.button("Set Budget Limit"):
                if check_api_health():
                    if api_set_privacy_limit(new_budget):
                        st.success(f"Budget limit set to ε = {new_budget}")
                        st.rerun()
                    else:
                        st.error("Failed to set budget via API")
                else:
                    tracker = get_privacy_tracker()
                    tracker.total_budget = new_budget
                    tracker._save_state()
                    st.success(f"Budget limit set to ε = {new_budget}")
                    st.rerun()
    
    with col2:
        with st.expander("Reset Privacy Budget", expanded=False):
            st.warning("⚠️ This will clear all privacy expenditure history!")
            if st.button("Reset Budget", type="secondary"):
                if check_api_health():
                    if api_reset_privacy():
                        st.success("Privacy budget reset successfully")
                        st.rerun()
                    else:
                        st.error("Failed to reset via API")
                else:
                    tracker = get_privacy_tracker()
                    tracker.reset()
                    st.success("Privacy budget reset successfully")
                    st.rerun()


def show_data_quality():
    st.header("📈 Data Quality Analysis")
    
    # Load data
    synth_data = load_synthetic_data()
    
    try:
        real_data = load_real_healthcare_data()
    except Exception as e:
        st.error(f"Failed to load real data: {e}")
        real_data = None
    
    if synth_data is None:
        st.warning("No synthetic data available. Please run generation first.")
        return
    
    if real_data is None:
        st.warning("Real data not available for comparison.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Statistics"])
    
    with tab1:
        st.subheader("Distribution Comparison")
        
        # Continuous features
        feature = st.selectbox("Select Feature", CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS)
        
        if feature in CONTINUOUS_COLUMNS:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=real_data[feature], name="Real", opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=synth_data[feature], name="Synthetic", opacity=0.7, nbinsx=30))
            fig.update_layout(barmode='overlay', title=f"Distribution: {feature}",
                              xaxis_title=feature, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Categorical
            real_counts = real_data[feature].value_counts(normalize=True)
            synth_counts = synth_data[feature].value_counts(normalize=True)
            
            df_compare = pd.DataFrame({
                'Real': real_counts,
                'Synthetic': synth_counts
            }).fillna(0)
            
            fig = px.bar(df_compare, barmode='group', title=f"Distribution: {feature}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Matrices")
        
        col1, col2 = st.columns(2)
        
        cont_cols = [c for c in CONTINUOUS_COLUMNS if c in real_data.columns]
        
        with col1:
            st.write("**Real Data**")
            real_corr = real_data[cont_cols].corr()
            fig = px.imshow(real_corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                           title="Real Data Correlation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Synthetic Data**")
            synth_cont = [c for c in cont_cols if c in synth_data.columns]
            synth_corr = synth_data[synth_cont].corr()
            fig = px.imshow(synth_corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                           title="Synthetic Data Correlation")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Build comparison table
        stats_data = []
        for col in CONTINUOUS_COLUMNS:
            if col in real_data.columns and col in synth_data.columns:
                stats_data.append({
                    'Feature': col,
                    'Real Mean': f"{real_data[col].mean():.2f}",
                    'Synth Mean': f"{synth_data[col].mean():.2f}",
                    'Mean Diff': f"{abs(real_data[col].mean() - synth_data[col].mean()):.2f}",
                    'Real Std': f"{real_data[col].std():.2f}",
                    'Synth Std': f"{synth_data[col].std():.2f}"
                })
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


def show_audit_log():
    st.header("📋 Audit Log")
    
    logs = get_logs()
    
    if not logs:
        st.info("No audit log entries yet.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        event_types = list(set(log.get('event_type', 'Unknown') for log in logs))
        selected_types = st.multiselect("Filter by Event Type", event_types, default=event_types)
    
    with col2:
        limit = st.slider("Show last N entries", 10, len(logs), min(50, len(logs)))
    
    # Filter and display
    filtered_logs = [log for log in logs if log.get('event_type', 'Unknown') in selected_types]
    filtered_logs = filtered_logs[-limit:][::-1]
    
    for log in filtered_logs:
        with st.expander(f"🔹 [{log.get('timestamp', '')[:19]}] {log.get('event_type', 'Unknown')}"):
            st.json(log.get('details', {}))


def show_training_history():
    st.header("⚙️ Training History")
    
    model_files = get_model_files()
    
    if not model_files:
        st.warning("No trained models found.")
        return
    
    # Model checkpoints
    st.subheader("Model Checkpoints")
    df = pd.DataFrame(model_files)
    df['modified'] = df['modified'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.dataframe(df, use_container_width=True)
    
    # Model size over rounds
    if len(model_files) > 1:
        fig = px.bar(df, x='round', y='size_kb', title="Model Size by Round")
        fig.update_layout(xaxis_title="Round", yaxis_title="Size (KB)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Latest model info
    st.subheader("Latest Model")
    latest = model_files[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Round", latest['round'])
    with col2:
        st.metric("Size", f"{latest['size_kb']:.1f} KB")
    with col3:
        st.metric("Last Modified", latest['modified'].strftime('%H:%M:%S'))


if __name__ == "__main__":
    main()
