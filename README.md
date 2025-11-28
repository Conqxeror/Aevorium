# Aevorium: Federated Synthetic Data Platform

A federated learning platform for training diffusion models on distributed sensitive healthcare data to produce high-fidelity synthetic datasets without raw data leaving source nodes. Built with Flower (FL framework), PyTorch, and Opacus (DP).

## ✨ Version 2.0 Features

### Core Capabilities
- ✅ **Expanded Schema** - 9 continuous + 5 categorical features from Synthea data
- ✅ **Enhanced Diffusion Model** - 6-layer ResNet with DDIM sampling
- ✅ **Privacy Budget Tracking** - Cumulative epsilon with API endpoints
- ✅ **QuantileTransformer** - Handles skewed healthcare utilization features
- ✅ **Named Model Checkpoints** - Prevents silent parameter mismatches
- ✅ **Categorical Softmax Sampling** - Distribution matching to real data

### Data Schema
**Continuous (9):** Age, BMI, BloodPressure, Glucose, EncounterCount, MedicationCount, ConditionCount, TotalCost, ProcedureCount

**Categorical (5):** Gender, Diagnosis, EncounterType, HasAllergies, RiskLevel

### Quick Start
See **[QUICK_START.md](QUICK_START.md)** for detailed instructions.

```powershell
# 1. Verify installation
python dry_run.py
python -m pytest -q

# 2. Start server (Terminal 1)
python server/server.py

# 3. Start 2+ clients (Terminal 2-3)
python node/client.py

# 4. Generate & validate (Terminal 4)
python generate_samples.py
python validate_data.py

# 5. Check privacy budget
curl http://localhost:8000/privacy-budget
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Aevorium Platform                        │
├─────────────────────────────────────────────────────────────┤
│  API (FastAPI :8000)                                        │
│  ├─ POST /train, /generate                                  │
│  ├─ GET /audit-log, /privacy-budget                         │
│  └─ POST /privacy-budget/set-limit, /reset                  │
├─────────────────────────────────────────────────────────────┤
│  Federation Server (Flower :8091)                           │
│  ├─ FedAvg aggregation                                      │
│  ├─ Named NPZ saves (encrypted)                             │
│  └─ Round metrics logging                                   │
├─────────────────────────────────────────────────────────────┤
│  Training Nodes (Opacus DP-SGD)                             │
│  ├─ TabularDiffusionModel (6 ResNet blocks)                 │
│  ├─ QuantileTransformer preprocessing                       │
│  └─ Privacy budget tracking                                 │
└─────────────────────────────────────────────────────────────┘
```

- **Server** (`server/`): Orchestrates FL rounds using Flower, saves encrypted models
- **Node** (`node/`): Local training client with Opacus DP, runs at data holder sites
- **Common** (`common/`): Shared model, preprocessing, schema, security, governance
- **API** (`api/`): FastAPI endpoints for training, generation, and monitoring
- **Data** (`data/csv/`): Synthea healthcare data files

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Federated Learning | Flower (flwr) | ≥1.5.0 |
| ML Framework | PyTorch | ≥2.0.0 |
| Diffusion Model | DDPM/DDIM | Custom |
| Differential Privacy | Opacus | ≥1.4.0 |
| API | FastAPI + Uvicorn | ≥0.100.0 |
| Preprocessing | scikit-learn | ≥1.3.0 |
| Encryption | cryptography (Fernet) | ≥41.0.0 |

## Documentation

📚 **Full documentation** available in `docs/` and `website/`:

### Essential Guides
- **[QUICK_START.md](QUICK_START.md)** - ⭐ Start here
- **[TRAINING_CONFIG.md](TRAINING_CONFIG.md)** - Configuration reference
- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - Changelog

### Architecture & Deployment  
- [Architecture Overview](docs/ARCHITECTURE.md) - System design, data flow
- [Getting Started](docs/GETTING_STARTED.md) - Installation and setup
- [Deployment Guide](docs/DEPLOYMENT.md) - Docker/Kubernetes

### API & Security
- [API Reference](docs/API_REFERENCE.md) - REST endpoints
- [Security & Privacy](docs/SECURITY_AND_PRIVACY.md) - DP guarantees, encryption
- [Validation Strategy](docs/VALIDATION.md) - Data quality metrics

### Monitoring
- [Dashboard Guide](docs/DASHBOARD.md) - Web UI monitoring dashboard

### View Docs Locally
```bash
cd website && npm install && npm start
```

## Getting Started

### Option 1: Local Development

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run validation:**
   ```powershell
   python dry_run.py
   python -m pytest -q
   ```

3. **Start training** (see [QUICK_START.md](QUICK_START.md)):
   ```powershell
   # Terminal 1: Server
   python server/server.py
   
   # Terminal 2-3: Clients (2+)
   python node/client.py
   
   # Terminal 4: Generate & Validate
   python generate_samples.py
   python validate_data.py
   ```

### Option 2: Docker (Production)
```powershell
./run_docker.ps1
# API at http://localhost:8000/docs
# Server at port 8091
```

### Option 3: Dashboard (Monitoring)
```powershell
# Start the web dashboard
streamlit run dashboard/app.py

# Access at http://localhost:8501
```

The dashboard provides:
- 📊 **Overview** - System health, training progress, privacy metrics
- 🚀 **Generate Data** - Create synthetic samples via UI
- 🔐 **Privacy Budget** - Track and manage ε consumption
- 📈 **Data Quality** - Compare real vs synthetic distributions
- 📋 **Audit Log** - Browse governance events
- ⚙️ **Training History** - View model checkpoints

See [Dashboard Guide](docs/DASHBOARD.md) for full documentation.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_EPOCHS` | 20 | Local epochs per round |
| `DP_NOISE_MULTIPLIER` | 0.3 | Privacy noise level |
| `NUM_ROUNDS` | 5 | FL training rounds |
| `SERVER_PORT` | 8091 | Federation server port |
| `BATCH_SIZE` | 32 | Training batch size |

## Output Files

| File | Description |
|------|-------------|
| `synthetic_data.csv` | Generated synthetic patients |
| `global_model_round_*.npz` | Encrypted model checkpoints |
| `preprocessor.joblib` | Fitted data preprocessor |
| `audit_log.json` | Governance audit trail |
| `privacy_budget.json` | Privacy expenditure tracking |

## License

MIT
