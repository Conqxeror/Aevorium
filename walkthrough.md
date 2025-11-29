# Walkthrough: Aevorium Phase 2 & 3

We have successfully upgraded Aevorium to a robust, containerized platform.

## Key Changes

### 1. Mixed-Type Data Support
- **Problem**: The initial PoC only supported continuous variables (Age, BMI). Real healthcare data has categories (Gender, Diagnosis).
- **Solution**: 
    - Implemented `DataPreprocessor` (One-Hot Encoding + Scaling).
    - Updated `TabularDiffusionModel` to handle dynamic input dimensions.
    - Defined a shared `schema.py`.

### 2. Model Robustness
- **Problem**: `BatchNorm1d` caused issues with Opacus (Differential Privacy) due to batch dependency.
- **Solution**: Replaced with `GroupNorm`, which is compatible with DP-SGD.

### 3. Dockerization
- **Problem**: Windows environment caused DLL issues with PyTorch/Opacus. Hard to reproduce.
- **Solution**: 
    - Created Dockerfiles for `Server`, `Node`, and `API`.
    - Created `docker-compose.yml` to orchestrate the full stack.
    - Centralized configuration in `common/config.py`.

## How to Run

### Option A: Docker (Recommended)
This runs the entire stack in isolated Linux containers, avoiding Windows DLL issues.

```powershell
./run_docker.ps1
```

### Option B: Local (Legacy)
If you cannot use Docker, you can still run the local script (might face DLL issues).

```powershell
./run_poc.ps1
```

## Verification Results

- **Code Integrity**: Verified via `dry_run.py`.
- **Docker Build**: Initiated and verified Docker availability.
- **Validation**: The `validate_data.py` script now produces plots for both continuous (histograms) and categorical (bar charts) variables.
