# Aevorium Dashboard

A comprehensive Streamlit-based monitoring dashboard for the Aevorium federated learning platform.

## Features

### 📊 Overview
- **System Health Monitoring**: Real-time API status indicator
- **Training Progress**: Track federated learning rounds and model checkpoints
- **Privacy Metrics**: Quick view of privacy budget consumption
- **Audit Activity**: Recent event log summary
- **Quality Score**: Overall synthetic data quality assessment

### 🚀 Generate Data
- **Sample Generation**: Configure number of samples to generate
- **Output Configuration**: Specify output filename
- **API Integration**: Generate via API when online, fallback to local generation
- **Live Preview**: View generated synthetic data preview

### 🔐 Privacy Budget
- **Cumulative Tracking**: Monitor total privacy epsilon spent
- **Budget Limits**: Set and manage privacy budget caps
- **Round History**: Visualize privacy expenditure over training rounds
- **Budget Reset**: Reset privacy counters when needed

### 📈 Data Quality
- **Distribution Comparison**: Compare real vs synthetic data distributions
- **Correlation Analysis**: Side-by-side correlation matrix visualization
- **Statistical Summary**: Detailed statistical comparison for all features
- **TVD Metrics**: Total Variation Distance for categorical features

### 📋 Audit Log
- **Event Filtering**: Filter by event type (MODEL_SAVED, privacy_budget_update, etc.)
- **Pagination**: Configurable entry limits
- **Detail View**: Expandable log entries with full JSON details

### ⚙️ Training History
- **Checkpoint List**: All model checkpoints with round, size, and timestamp
- **Size Visualization**: Model size progression chart
- **Latest Model Info**: Quick access to current model details

## Getting Started

### Prerequisites
- Python 3.10+
- Streamlit >= 1.28.0
- Plotly >= 5.18.0

### Installation

```bash
pip install streamlit plotly requests
```

### Running the Dashboard

```bash
# From project root
streamlit run dashboard/app.py

# Or with custom port
streamlit run dashboard/app.py --server.port 8502
```

### Configuration

The dashboard uses environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `AEVORIUM_API_URL` | `http://localhost:8000` | Base URL for the Aevorium API |

## Dashboard Pages

### Overview Page
The main landing page provides a quick system health check:

- **Training Rounds**: Number of completed FL rounds
- **Synthetic Samples**: Count of generated samples
- **Privacy ε Spent**: Cumulative privacy budget consumption
- **Audit Events**: Total logged events
- **Recent Activity**: Last 5 audit log entries

### Generate Data Page
Controls for synthetic data generation:

1. **Number of Samples**: Spinner to select sample count (100-100,000)
2. **Output Filename**: Text input for output CSV name
3. **Generate Button**: Triggers generation via API or locally
4. **Data Preview**: Shows first rows of generated data

### Privacy Budget Page
Detailed privacy tracking:

- **Summary Metrics**:
  - Cumulative epsilon spent
  - Remaining budget (if limit set)
  - Percentage used

- **History Chart**: Line chart showing epsilon accumulation
- **Configuration Panel**:
  - Set budget limit
  - Reset privacy counters

### Data Quality Page
Synthetic data validation:

- **Quality Score Card**: Overall quality metric (0-100%)
- **Distribution Plots**: Side-by-side histograms for continuous features
- **Correlation Heatmaps**: Real vs synthetic correlation matrices
- **Statistical Table**: Mean, std comparisons per feature

### Audit Log Page
Event browsing:

- **Event Type Filter**: Multi-select for event types
- **Entry Limit**: Slider for pagination
- **Expandable Details**: Click to view full event JSON

### Training History Page
Model checkpoint management:

- **Checkpoint Table**: File, round, size, modification time
- **Size Chart**: Bar chart of model sizes by round
- **Latest Model Card**: Quick reference to current model

## Auto-Refresh

Enable auto-refresh in the sidebar to automatically update the dashboard every 30 seconds. Useful for monitoring active training sessions.

## API Integration

The dashboard integrates with the Aevorium API for:

- **Health Check**: `/health` endpoint for status
- **Generation**: `/generate` endpoint for sample creation
- **Privacy Budget**: `/privacy-budget` endpoints for tracking/management

When the API is offline, the dashboard falls back to local file-based operations where possible.

## Troubleshooting

### Dashboard Won't Start
```bash
# Check Streamlit installation
pip show streamlit

# Reinstall if needed
pip install --force-reinstall streamlit
```

### API Offline Warning
The dashboard shows "API Offline" when it cannot reach the Aevorium API. This is normal if:
- The API server isn't running
- The API is on a different host/port

To start the API:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### No Synthetic Data
If "No synthetic data available" appears:
1. Train a model first (`run_poc.ps1` or Docker)
2. Generate samples via the Generate Data page
3. Or run `python generate_samples.py`

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## Screenshots

### Overview Dashboard
![Overview](../assets/dashboard-overview.png)

### Privacy Budget Tracking
![Privacy](../assets/dashboard-privacy.png)

### Data Quality Analysis
![Quality](../assets/dashboard-quality.png)

## Version History

| Version | Changes |
|---------|---------|
| 2.0.0 | API integration, generation controls, enhanced styling |
| 1.0.0 | Initial release with basic monitoring |
