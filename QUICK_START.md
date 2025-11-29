# Quick Start Guide - Aevorium v2.0

## What's New in v2.0

### 1. Expanded Data Schema
- **9 continuous features**: Age, BMI, BloodPressure, Glucose + healthcare utilization metrics
- **5 categorical features**: Gender, Diagnosis, EncounterType, HasAllergies, RiskLevel
- **Real data integration**: Loads from Synthea CSV files in `data/csv/`

### 2. Enhanced Model Architecture
- Deeper network (6 residual blocks, hidden dim 512)
- Cosine noise schedule for better sample quality
- DDIM sampling (200 steps) for faster generation
- Huber loss for robustness to outliers

### 3. Improved Preprocessing
- QuantileTransformer for skewed features (costs, counts)
- StandardScaler for clinical features
- Persistent preprocessor saved to `preprocessor.joblib`

### 4. Privacy Budget Tracking
- Cumulative epsilon tracking across rounds
- API endpoints for budget management
- Per-round privacy expenditure logging

### 5. Named Model Saves
- Parameter names in NPZ files prevent silent mismatches
- Backward compatible with positional saves

### 6. Categorical Softmax Sampling
- Per-category softmax with temperature control
- Distribution matching to real data
- All categories properly represented

## Quick Validation

Before training, verify installation:

```powershell
# Test imports and preprocessing
python dry_run.py

# Run test suite (should pass)
python -m pytest -q
```

## Running Training

### Option 1: Quick Test (Fast, No Privacy)
```powershell
# Terminal 1: Server
python server/server.py

# Terminal 2-3: Clients (run 2+)
$env:TRAINING_EPOCHS="5"
$env:DP_NOISE_MULTIPLIER="0.0"
python node/client.py

# Terminal 4: Generate & Validate
python generate_samples.py
python validate_data.py
```

### Option 2: Balanced Training (Default)
```powershell
# Terminal 1: Server
python server/server.py

# Terminal 2-3: Clients (run 2+)
# Uses defaults: 20 epochs, 0.3 noise
python node/client.py

# Terminal 4: Generate & Validate
python generate_samples.py
python validate_data.py
```

### Option 3: High Quality Training (More Epochs)
```powershell
# Terminal 1: Server
$env:NUM_ROUNDS="10"
python server/server.py

# Terminal 2-3: Clients (run 2+)
$env:TRAINING_EPOCHS="30"
$env:DP_NOISE_MULTIPLIER="0.2"
python node/client.py

# Terminal 4: Generate & Validate
python generate_samples.py
python validate_data.py
```

## Expected Results

### Quality Metrics
With proper training, you should see:
- All 9 continuous features within valid ranges
- All 5 categorical features with proper distributions
- Quality score > 80% from `validate_data.py`
- Correlation differences < 0.15

### Example Synthetic Data Output
| Feature | Expected Range |
|---------|---------------|
| Age | 18-90 years |
| BMI | 15-50 |
| EncounterCount | 0-500 |
| TotalCost | $0-$300,000 |
| Gender | Male/Female/Other |
| Diagnosis | All 6 categories |
| RiskLevel | Low/Medium/High |

## Monitoring Training

### Web Dashboard (Recommended)
```powershell
# Start the Streamlit dashboard
streamlit run dashboard/app.py

# Access at http://localhost:8501
```

The dashboard provides real-time monitoring of:
- Training progress and model checkpoints
- Privacy budget consumption
- Synthetic data quality metrics
- Audit log browsing
- On-demand sample generation

### View privacy budget:
```powershell
# Via API (if running)
curl http://localhost:8000/privacy-budget

# Or check file directly
cat privacy_budget.json
```

### Check model files:
```powershell
ls global_model_round_*.npz
```

### View audit log:
```powershell
cat audit_log.json | Select-Object -Last 20
```

## Troubleshooting

### Still getting poor quality?
1. **Train longer**: Set `$env:TRAINING_EPOCHS="30"`
2. **Lower noise**: Set `$env:DP_NOISE_MULTIPLIER="0.2"`
3. **More rounds**: Set `$env:NUM_ROUNDS="10"` before starting server
4. **More clients**: Run 3-4 clients instead of 2

### Training too slow?
1. Reduce epochs: `$env:TRAINING_EPOCHS="10"`
2. Reduce batch size: `$env:BATCH_SIZE="16"`
3. Enable GPU if available

### Out of memory?
1. Reduce batch size: `$env:BATCH_SIZE="16"` or `8`
2. Reduce workers: `$env:DATALOADER_NUM_WORKERS="0"`
3. Use CPU: The code auto-detects GPU, no changes needed

### Privacy budget exhausted?
1. Check status: `GET /privacy-budget`
2. Reset if needed: `POST /privacy-budget/reset`

## Next Steps

1. Run tests: `python -m pytest -q`
2. Start training cycle (see Option 2 above)
3. Validate results: `python validate_data.py`
4. Check privacy: `curl http://localhost:8000/privacy-budget`
5. Generate more samples: `python generate_samples.py`

## Key Files

| File | Purpose |
|------|---------|
| `server/server.py` | Federation server (port 8091) |
| `node/client.py` | Training client with DP |
| `generate_samples.py` | Sample generation with DDIM |
| `validate_data.py` | Quality score calculation |
| `api/main.py` | REST API (port 8000) |

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_EPOCHS` | 20 | Epochs per FL round |
| `DP_NOISE_MULTIPLIER` | 0.3 | Privacy noise level |
| `DP_MAX_GRAD_NORM` | 1.0 | Gradient clipping |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_ROUNDS` | 5 | FL rounds (server) |
| `SERVER_PORT` | 8091 | Server port |

## Performance Benchmarks

| Configuration | Training Time | Quality | Privacy (ε) |
|--------------|---------------|---------|-------------|
| Fast (5 epochs, no DP) | ~2 min | ~85% | None |
| Default (20 epochs, 0.3 noise) | ~8 min | ~80% | ~40-50 |
| High Quality (30 epochs, 0.2) | ~12 min | ~85% | ~60-70 |

*Times for 2 clients, 500 samples each, CPU*
