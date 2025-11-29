# Getting Started with Aevorium

## Prerequisites

- Python 3.9+
- pip
- PowerShell (for Windows scripts)
- CUDA-capable GPU (optional, for faster training)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd Aevorium
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
    **For GPU Support** (NVIDIA CUDA):
    ```powershell
    pip uninstall -y torch torchvision torchaudio
    pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio --upgrade
    ```

3.  **Quick Validation** (verify installation):
    ```bash
    python dry_run.py
    python -m pytest -q
    ```

## Running with Docker (Recommended for Production)

We provide a helper script to run the full stack using Docker Compose.

1.  **Start the Stack**:
    ```powershell
    ./run_docker.ps1
    ```
    
    **What happens?**
    - Builds Docker images for Server, Node, and API
    - Starts the Federation Server (port 8091)
    - Starts 2 Training Nodes
    - Starts the API (port 8000)
    - A shared volume persists the model and logs

2.  **Access Components**:
    - **API Docs**: `http://localhost:8000/docs`
    - **Audit Log**: `http://localhost:8000/audit-log`
    - **Privacy Budget**: `http://localhost:8000/privacy-budget`

3.  **Stop the Stack**:
    ```powershell
    docker-compose down
    ```

## Running Locally (Recommended for Development)

### Quick Start

1.  **Terminal 1 - Start Server**:
    ```powershell
    python server/server.py
    ```
    Server listens on port 8091 by default.

2.  **Terminal 2 & 3 - Start Clients** (need at least 2):
    ```powershell
    python node/client.py
    ```

3.  **Terminal 4 - Generate & Validate** (after training completes):
    ```powershell
    python generate_samples.py
    python validate_data.py
    ```

### With Custom Configuration

Use environment variables to customize training:

```powershell
# High quality training (more epochs, less noise)
$env:TRAINING_EPOCHS="20"
$env:DP_NOISE_MULTIPLIER="0.3"
python node/client.py

# Fast debugging (fewer epochs, no privacy)
$env:TRAINING_EPOCHS="5"
$env:DP_NOISE_MULTIPLIER="0.0"
python node/client.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 8091 | Federation server port |
| `SERVER_ADDRESS` | 127.0.0.1:8091 | Server address for clients |
| `NUM_ROUNDS` | 5 | Number of FL rounds |
| `TRAINING_EPOCHS` | 20 | Local epochs per round |
| `DP_NOISE_MULTIPLIER` | 0.3 | Differential privacy noise |
| `DP_MAX_GRAD_NORM` | 1.0 | Gradient clipping norm |
| `BATCH_SIZE` | 32 | Training batch size |
| `USE_AMP` | 0 | Enable mixed precision (0/1) |

## Running the API

To interact with the system programmatically:

1.  **Start the API Server**:
    ```bash
    python api/main.py
    ```
    The server runs on `http://0.0.0.0:8000`.

2.  **Test the API**:
    ```bash
    python test_api.py
    ```
    Or visit the interactive docs at `http://127.0.0.1:8000/docs`.

## Output Files

After training and generation:

| File | Description |
|------|-------------|
| `synthetic_data.csv` | Generated synthetic patient data |
| `global_model_round_X.npz` | Encrypted model checkpoints |
| `preprocessor.joblib` | Fitted data preprocessor |
| `audit_log.json` | Governance audit trail |
| `privacy_budget.json` | Privacy expenditure tracking |
| `validation_plots_*.png` | Distribution comparison plots |

## Running Tests

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_sampling.py

# Run with coverage
python -m pytest --cov=common
```

## Troubleshooting

### Server Connection Issues
- Ensure server is running before starting clients
- Check port 8091 is not in use
- Verify `SERVER_ADDRESS` environment variable if using custom port

### Poor Synthetic Data Quality
1. Increase training epochs: `$env:TRAINING_EPOCHS="20"`
2. Lower DP noise: `$env:DP_NOISE_MULTIPLIER="0.3"`
3. Run more FL rounds: Set `NUM_ROUNDS` higher in server

### Out of Memory
1. Reduce batch size: `$env:BATCH_SIZE="16"`
2. Use CPU: The code auto-detects and uses CPU if no GPU available

### Missing Model File
- Ensure training completed successfully
- Check `global_model_round_*.npz` files exist in project root
