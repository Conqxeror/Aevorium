# Training Configuration Reference

## Environment Variables

### Training Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_EPOCHS` | 20 | Number of local training epochs per round |
| `DP_NOISE_MULTIPLIER` | 0.3 | Differential privacy noise level (lower = less privacy, better quality) |
| `DP_MAX_GRAD_NORM` | 1.0 | Maximum gradient norm for clipping |
| `BATCH_SIZE` | 32 | Training batch size |
| `USE_AMP` | 0 | Enable mixed precision training (0 or 1) |
| `DATALOADER_NUM_WORKERS` | 4 | Number of DataLoader worker threads |

### Server Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 8091 | Federation server port |
| `SERVER_ADDRESS` | 127.0.0.1:8091 | Server address (for clients) |
| `NUM_ROUNDS` | 5 | Number of federated learning rounds |

### Examples

**Fast debugging (no privacy, minimal epochs):**
```powershell
$env:TRAINING_EPOCHS="5"
$env:DP_NOISE_MULTIPLIER="0.0"
python node/client.py
```

**Balanced training (default):**
```powershell
# Uses defaults: 20 epochs, 0.3 noise multiplier
python node/client.py
```

**High privacy training:**
```powershell
$env:DP_NOISE_MULTIPLIER="1.0"
$env:TRAINING_EPOCHS="10"
python node/client.py
```

**High quality training (more epochs, less noise):**
```powershell
$env:TRAINING_EPOCHS="30"
$env:DP_NOISE_MULTIPLIER="0.2"
python node/client.py
```

## Data Schema

### Continuous Features (9 total)
Features are clipped to medically plausible ranges after generation:

| Feature | Range | Description |
|---------|-------|-------------|
| Age | 18-90 | Patient age in years |
| BMI | 15-50 | Body Mass Index (kg/m²) |
| BloodPressure | 80-200 | Systolic blood pressure (mmHg) |
| Glucose | 50-400 | Blood glucose (mg/dL) |
| EncounterCount | 0-500 | Total healthcare encounters |
| MedicationCount | 0-100 | Number of medications |
| ConditionCount | 0-30 | Number of conditions |
| TotalCost | 0-300,000 | Total healthcare costs ($) |
| ProcedureCount | 0-200 | Number of procedures |

### Categorical Features (5 total)

| Feature | Categories |
|---------|------------|
| Gender | Male, Female, Other |
| Diagnosis | Diabetes, Hypertension, Healthy, HeartDisease, ChronicKidneyDisease, Cancer |
| EncounterType | wellness, ambulatory, outpatient, urgentcare, emergency, inpatient |
| HasAllergies | Yes, No |
| RiskLevel | Low, Medium, High |

### Total Input Dimension
- **Continuous**: 9 features
- **Categorical (one-hot)**: 3 + 6 + 6 + 2 + 3 = 22 features
- **Total**: 31 input dimensions

## Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input dimension | 31 | Total features (continuous + one-hot) |
| Hidden dimension | 512 | Width of MLP layers |
| Time embedding | 128 | Sinusoidal time embedding size |
| Depth | 6 | Number of residual blocks |
| Dropout | 0.1 | Dropout probability |
| Diffusion timesteps | 1500 | Number of noise steps |
| Noise schedule | Cosine | Nichol & Dhariwal schedule |

### Sampling Configuration
- **Method**: DDIM (Denoising Diffusion Implicit Models)
- **DDIM Steps**: 200 (faster than full 1500 DDPM steps)
- **Eta**: 0.3 (stochasticity: 0=deterministic, 1=DDPM)
- **Categorical Temperature**: 0.2 (sharper distributions)

## Preprocessing

### Normal Features (StandardScaler)
- Age, BMI, BloodPressure, Glucose

### Skewed Features (QuantileTransformer)
Maps to normal distribution to handle extreme values:
- EncounterCount, MedicationCount, ConditionCount, TotalCost, ProcedureCount

### Categorical (OneHotEncoder)
- Predefined categories from schema
- `handle_unknown='ignore'` for robustness

## Privacy Guarantees

### Privacy Budget Tracking
The system tracks cumulative epsilon (ε) across training rounds:

```bash
# Check current privacy status
curl http://localhost:8000/privacy-budget

# Set a total budget limit
curl -X POST http://localhost:8000/privacy-budget/set-limit \
  -H "Content-Type: application/json" \
  -d '{"total_budget": 50.0}'
```

### Typical Privacy-Utility Tradeoffs

| Configuration | ε per round | Total ε (5 rounds) | Quality |
|--------------|-------------|-------------------|---------|
| noise=1.0, epochs=10 | ~2-3 | ~10-15 | Lower |
| noise=0.5, epochs=20 | ~5-8 | ~25-40 | Moderate |
| noise=0.3, epochs=20 | ~8-12 | ~40-60 | Higher |
| noise=0.0 (no DP) | 0 | 0 | Best (no privacy) |

## Troubleshooting

### Poor synthetic data quality
1. Increase `TRAINING_EPOCHS` to 30+
2. Reduce `DP_NOISE_MULTIPLIER` to 0.2-0.3
3. Run more federation rounds (increase `NUM_ROUNDS`)
4. Ensure 2+ clients are training

### Training too slow
1. Reduce `TRAINING_EPOCHS` to 10
2. Use smaller batch sizes: `$env:BATCH_SIZE="16"`
3. Reduce diffusion timesteps (edit `model.py`)
4. Enable GPU acceleration

### Out of memory
1. Reduce `BATCH_SIZE` to 16 or 8
2. Reduce `DATALOADER_NUM_WORKERS` to 0
3. Use CPU instead of GPU for small datasets

### Privacy budget exhausted
1. Check current budget: `GET /privacy-budget`
2. Reset if appropriate: `POST /privacy-budget/reset`
3. Use lower noise multiplier for remaining rounds

## Performance Benchmarks

| Configuration | Training Time | Quality Score | Privacy (ε) |
|--------------|---------------|--------------|-------------|
| Fast (5 epochs, no DP) | ~2 min | ~85% | None |
| Balanced (20 epochs, 0.3 noise) | ~8 min | ~80% | ~40-50 |
| High Quality (30 epochs, 0.2 noise) | ~12 min | ~85% | ~60-70 |
| High Privacy (10 epochs, 1.0 noise) | ~4 min | ~60% | ~10-15 |

*Times for 2 clients, 500 samples each, on CPU*
