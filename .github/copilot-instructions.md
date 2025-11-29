# Aevorium Copilot Instructions

## Project Overview
Aevorium is a **federated learning platform** for training diffusion models on distributed healthcare data to produce synthetic datasets without raw data leaving source nodes. Built with Flower (FL framework), PyTorch, and FastAPI.

## Architecture (Hub-and-Spoke)

| Component | Location | Purpose |
|-----------|----------|---------|
| Federation Server | `server/server.py` | Orchestrates FL rounds using Flower, aggregates model weights |
| Training Node | `node/client.py` | Local training with Opacus (DP-SGD), runs at data holder sites |
| Common | `common/` | Shared model (`model.py`), preprocessing (`preprocessing.py`), schema (`schema.py`) |
| API | `api/main.py` | FastAPI endpoints for `/train`, `/generate`, `/audit-log` |
| Infra | `infra/`, `docker-compose.yml` | Dockerized deployment for all components |

## Critical Patterns

### Data Schema (`common/schema.py`)
Schema is consortium-agreed and must stay consistent across all clients:
```python
CONTINUOUS_COLUMNS = ['Age', 'BMI', 'BloodPressure', 'Glucose']
CATEGORICAL_COLUMNS = ['Gender', 'Diagnosis']
CATEGORIES = {'Gender': ['Male', 'Female', 'Other'], 'Diagnosis': [...]}
```
**Always** update `get_input_dim()` when modifying schema—it computes total dimensions after one-hot encoding.

### Model Architecture (`common/model.py`)
- `TabularDiffusionModel`: DDPM-style MLP with sinusoidal time embeddings
- Uses `GroupNorm` (not BatchNorm) for Opacus compatibility
- `DiffusionManager`: Handles noise scheduling, `add_noise()`, `train_step()`, `sample()`

### Privacy (Opacus Integration in `node/client.py`)
Clients wrap models with `PrivacyEngine.make_private()` each training round. Key pattern:
```python
training_model = copy.deepcopy(self.model)  # Avoid double-wrapping
model, optimizer, train_loader = privacy_engine.make_private(...)
```

### Security (`common/security.py`)
Model checkpoints are Fernet-encrypted at rest. Use `decrypt_file()` to load, `encrypt_file()` after saving.

## Developer Workflows

### Quick Validation (No Docker)
```powershell
python dry_run.py  # Tests imports, preprocessing, model forward pass
```

### Local PoC (Legacy)
```powershell
./run_poc.ps1  # Starts server + 2 clients, generates samples, validates
```

### Docker (Recommended)
```powershell
./run_docker.ps1  # Builds and runs full stack
# API at http://localhost:8000/docs, Server at :8080
docker-compose down  # Cleanup
```

### Test API Endpoints
```powershell
python test_api.py  # Requires API running
```

### Validate Synthetic Data Quality
```powershell
python validate_data.py  # Compares real vs synthetic distributions
```

## Key Conventions

1. **Paths**: Use `common/config.py` for all file paths (`AUDIT_LOG_FILE`, `MODEL_DIR`, etc.)
2. **Logging**: All critical events → `log_event()` in `common/governance.py` → `audit_log.json`
3. **Model Checkpoints**: Saved as `global_model_round_X.npz` (encrypted), loaded via `load_latest_model_weights()` in `generate_samples.py`
4. **Preprocessing**: Always fit `DataPreprocessor` before transform—inverse transform reconstructs original schema

## Common Pitfalls

- **Input dimension mismatch**: Model expects `get_input_dim()` features (continuous + one-hot encoded categorical)
- **Opacus double-wrap**: Always deepcopy model before `make_private()` each round
- **Encrypted model loading**: Must call `decrypt_file()` then load from `io.BytesIO()`
- **Docker paths**: Override `STORAGE_DIR` env var for container volume mounts

## Documentation
- Architecture: `docs/ARCHITECTURE.md`
- API Reference: `docs/API_REFERENCE.md`
- Docusaurus site: `cd website && npm start`

## Note:
- Never run docker, until and unless specifically told to do, try to run everything locally.