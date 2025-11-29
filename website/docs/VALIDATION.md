# Validation & Quality Assurance

Ensuring the synthetic data is useful (high utility) and safe (high privacy) is critical.

## Validation Pipeline (`validate_data.py`)

We automatically run a suite of statistical tests comparing the **Synthetic Data** against **Ground Truth** (real or reference data).

```bash
python validate_data.py
```

## Metrics Computed

### 1. Continuous Features

**Features Validated (9 total):**
| Feature | Range | Unit |
|---------|-------|------|
| Age | 18-90 | years |
| BMI | 15-50 | kg/m² |
| BloodPressure | 80-200 | mmHg |
| Glucose | 50-400 | mg/dL |
| EncounterCount | 0-500 | count |
| MedicationCount | 0-100 | count |
| ConditionCount | 0-30 | count |
| TotalCost | 0-300,000 | $ |
| ProcedureCount | 0-200 | count |

**Metrics:**
- **Mean Comparison**: Real vs Synthetic mean values
- **Standard Deviation Comparison**: Variance preservation
- **Range Validation**: All values within expected clinical ranges
- **Distribution Shape**: Visual histogram comparison

### 2. Categorical Features

**Features Validated (5 total):**
| Feature | Categories |
|---------|------------|
| Gender | Male, Female, Other |
| Diagnosis | Diabetes, Hypertension, Healthy, HeartDisease, ChronicKidneyDisease, Cancer |
| EncounterType | wellness, ambulatory, outpatient, urgentcare, emergency, inpatient |
| HasAllergies | Yes, No |
| RiskLevel | Low, Medium, High |

**Metrics:**
- **Frequency Comparison**: Category proportions in real vs synthetic
- **Total Variation Distance (TVD)**: $TVD = \frac{1}{2} \sum_i |P_{real}(i) - P_{synth}(i)|$
- **Category Coverage**: All categories represented in synthetic data

### 3. Correlation Analysis

- **Metric**: Pearson Correlation Matrix difference for continuous variables
- **Goal**: Preserve relationships (e.g., Age-EncounterCount, BMI-Glucose)
- **Calculation**: $\text{Diff} = \text{mean}(|Corr_{Real} - Corr_{Synth}|)$
- **Target**: Difference < 0.15 indicates good structure preservation

## Quality Score Calculation

The validation script computes an overall quality score:

```
QUALITY SCORE = 0.5 × Continuous Score + 0.5 × Categorical Score
```

### Continuous Score (per feature)
```
score = 1.0 - min(1.0, 0.6 × mean_error + 0.4 × std_error)
```
Where:
- `mean_error = |real_mean - synth_mean| / feature_range`
- `std_error = |real_std - synth_std| / real_std`

### Categorical Score (per feature)
```
score = 1.0 - min(1.0, TVD)
```

### Example Output
```
======================================================================
QUALITY SCORE SUMMARY
======================================================================
Continuous Features Score: 87.3%
  Age: 92.1%
  BMI: 89.5%
  BloodPressure: 85.2%
  ...

Categorical Features Score: 91.8%
  Gender: 95.2%
  Diagnosis: 88.4%
  ...

OVERALL QUALITY SCORE: 89.6%
======================================================================
```

## Visualization

The script generates comparison plots:

### `validation_plots_continuous.png`
- Overlaid histograms (Real vs Synthetic) for each continuous feature
- Density-normalized for fair comparison

### `validation_plots_categorical.png`
- Side-by-side bar charts showing category frequencies
- Easy visual detection of missing/overrepresented categories

## Interpreting Results

### Good Result (Quality > 80%)
- Means within 10% of real data
- All histogram shapes similar
- All categories represented
- Correlation difference < 0.15

### Moderate Result (60-80%)
- Some feature means off by 10-20%
- Minor category imbalances
- Acceptable for exploratory analysis

### Poor Result (< 60%)
- **Mode collapse**: Single value dominates
- **Missing categories**: Some categories never generated
- **Extreme values**: Values outside clinical ranges

## Improving Quality

If validation shows poor quality:

1. **Increase Training**:
   ```powershell
   $env:TRAINING_EPOCHS="30"
   $env:NUM_ROUNDS="10"
   ```

2. **Reduce DP Noise** (trades privacy for quality):
   ```powershell
   $env:DP_NOISE_MULTIPLIER="0.2"
   ```

3. **Add More Clients**: More data sources improve model

4. **Check Preprocessor**: Ensure `preprocessor.joblib` exists from training

## Privacy Validation

### Theoretical Guarantees
- Differential Privacy (DP-SGD) provides mathematical bounds
- Check `privacy_budget.json` for cumulative epsilon

### Empirical Validation (Future)
- Membership Inference Attacks (MIA)
- Attribute Inference Attacks
- Linkage attacks against known records

## Running Full Validation Pipeline

```bash
# Generate samples
python generate_samples.py

# Run validation
python validate_data.py

# Check privacy budget
curl http://localhost:8000/privacy-budget

# View audit log
cat audit_log.json | Select-Object -Last 10
```
