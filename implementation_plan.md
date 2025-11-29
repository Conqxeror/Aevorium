# Implementation Plan - Phase 2: Core Model & Infrastructure Hardening

# Goal Description
The current PoC proves the concept but lacks the robustness required for real-world healthcare data (which is mixed-type: continuous + categorical) and scalable deployment. This phase aims to upgrade the **Diffusion Model** to handle categorical data and **Dockerize** the application for easier deployment.

## User Review Required
> [!IMPORTANT]
> **Breaking Change**: The model architecture will change to support embeddings. Old checkpoints (`.npz`) will be incompatible.
> **Dependency**: Will require `pandas` and `sklearn` (for preprocessing) in the Node environment.

## Proposed Changes

### 1. Data Handling (`common/data.py`, `node/client.py`)
- **Current**: Generates synthetic float-only data.
- **New**: 
    - Add support for loading external CSV files.
    - Implement a `Preprocessor` class to handle:
        - Normalization for continuous columns.
        - One-Hot or Embedding encoding for categorical columns.
    - Update `TabularDataset` to return mixed tensors (or concatenated vector).

### 2. Model Architecture (`common/model.py`)
- **Current**: Simple MLP expecting fixed input dimension.
- **New**: 
    - Add `Embedding` layers for categorical inputs.
    - Concatenate embeddings with continuous inputs before the main MLP.
    - Update `DiffusionManager` to handle mixed data types (sampling categorical requires argmax or special handling).

### 3. Infrastructure (`infra/`)
- **Current**: `Dockerfile.node` exists but is basic. No server Dockerfile.
- **New**:
    - `infra/Dockerfile.server`: Container for Federation Server.
    - `infra/Dockerfile.api`: Container for FastAPI.
    - `docker-compose.yml`: Orchestrate Server, 2 Nodes, and API.

### 4. Orchestration (`api/main.py`)
- **Current**: Calls `run_poc.ps1`.
- **New**: 
    - Use `docker-compose up` or Python Docker SDK to spawn training jobs dynamically.

## Verification Plan

### Automated Tests
- **Unit Tests**: Test `Preprocessor` on mixed data.
- **Integration**: Run `docker-compose up` and verify full training cycle (3 rounds).
- **Validation**: Use `validate_data.py` on the new mixed-type synthetic data.

### Manual Verification
- Inspect `audit_log.json` to ensure Dockerized components are logging correctly.
