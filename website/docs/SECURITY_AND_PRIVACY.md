# Security and Privacy

Aevorium is built with a "Privacy by Design" approach, ensuring that sensitive healthcare data remains secure and private throughout the lifecycle.

## 1. Federated Learning (Data Minimization)
- **Concept**: Instead of moving data to a central server, we move the model to the data.
- **Implementation**: We use **Flower (flwr)** with FedAvg strategy. The server sends the model weights to the clients. Clients train locally on their private datasets. Only the *weight updates* (gradients) are sent back to the server.
- **Benefit**: Raw patient data (EHR, genomics) never leaves the hospital firewall.

## 2. Differential Privacy (DP)

### Concept
Adds mathematical noise to the training process so that the model cannot memorize or regurgitate any single individual's data.

### Implementation
We use **Opacus** (by Meta) to wrap the PyTorch optimizer:

- **Gradient Clipping**: Per-sample gradients are clipped to a maximum norm ($C$) to bound the influence of any single record.
- **Noise Addition**: Gaussian noise ($\sigma$) is added to the summed gradients.
- **Configuration**:
    ```powershell
    $env:DP_NOISE_MULTIPLIER="0.3"  # Noise level (higher = more privacy)
    $env:DP_MAX_GRAD_NORM="1.0"     # Clipping norm
    ```

### Privacy Budget Tracking

The system includes a comprehensive **Privacy Budget Tracker** (`PrivacyBudgetTracker` class) that:

1. **Tracks Cumulative Epsilon**: Records privacy expenditure across all training rounds
2. **Enforces Budget Limits**: Training fails if cumulative ε exceeds configured limit
3. **Provides Visibility**: 
    - Per-round epsilon values
    - Remaining budget calculations
    - Round history with full details
4. **Persists State**: Saves to `privacy_budget.json` for durability

#### API Endpoints
```bash
# Check current privacy status
GET /privacy-budget

# Set a total budget limit (e.g., ε = 50)
POST /privacy-budget/set-limit
{"total_budget": 50.0, "delta": 1e-5}

# Reset budget (WARNING: clears history)
POST /privacy-budget/reset
```

#### Example Privacy Budget Response
```json
{
  "cumulative_epsilon": 15.6,
  "delta": 1e-5,
  "total_budget": 50.0,
  "budget_remaining": 34.4,
  "budget_exhausted_pct": 31.2,
  "num_rounds": 3,
  "avg_epsilon_per_round": 5.2,
  "round_history": [...]
}
```

### Composition
- Uses **simple composition**: total ε = sum of per-round ε values
- Future: Advanced composition (RDP/zCDP) for tighter bounds

### Typical Privacy Guarantees

| Configuration | Epsilon (ε) | Privacy Level |
|--------------|-------------|---------------|
| noise=1.0, epochs=10, 3 rounds | ~8-10 | Strong |
| noise=0.5, epochs=20, 5 rounds | ~25-30 | Moderate |
| noise=0.3, epochs=20, 5 rounds | ~40-50 | Lower (higher utility) |

## 3. Model Encryption (Security at Rest)

### Concept
The global model contains aggregated knowledge from all hospitals. While differentially private, it's a valuable asset requiring protection.

### Implementation
- **Algorithm**: **Fernet** (symmetric encryption) from the `cryptography` library
- **Key Management**: A `secret.key` is generated automatically on first run
- **Process**:
    1. Server aggregates updates after each round
    2. Server saves `.npz` file with named parameters
    3. Server immediately encrypts the file on disk
- **Access**: The API/Generator must possess the key to decrypt and load the model

### Named Model Saves
Models are saved with parameter names (not positional indices) to prevent silent mismatches:
```python
# Safe: Named save
np.savez(file, **{"layer1.weight": w1, "layer1.bias": b1, ...})

# Risky (old approach): Positional save
np.savez(file, w1, b1, ...)  # Order changes = wrong weights loaded
```

## 4. Governance & Auditing

### Concept
Traceability is crucial for healthcare compliance (HIPAA/GDPR).

### Implementation
- **Audit Log**: JSON log in `audit_log.json` records every critical action
- **Privacy Log**: Separate `privacy_budget.json` for detailed privacy tracking
- **Events Logged**:
    | Event | Description |
    |-------|-------------|
    | `TRAINING_STARTED` | FL session initiated |
    | `MODEL_SAVED` | Checkpoint created and encrypted |
    | `ROUND_METRICS` | Per-round aggregated loss |
    | `DATA_GENERATION` | Synthetic data requested |
    | `privacy_budget_update` | Epsilon spent this round |
    | `privacy_budget_exceeded` | Budget limit hit |

### Example Audit Entry
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-28T10:30:00.000000",
  "event_type": "privacy_budget_update",
  "details": {
    "client_id": "0",
    "round_num": 3,
    "epsilon_spent": 5.2,
    "cumulative_epsilon": 15.6,
    "budget_limit": 50.0
  }
}
```

## 5. Secure Data Handling

### Data Sources
- **Real Data**: Loaded from `data/csv/` (Synthea format)
- **Reference Data**: Fallback synthetic data for testing

### Preprocessing Security
- Preprocessor (`DataPreprocessor`) is fitted on local data only
- Saved to `preprocessor.joblib` for consistent transforms
- No raw data statistics leave the client

### Generated Data
- Synthetic data output contains no direct patient identifiers
- DP guarantees limit information leakage about training individuals

## 6. Future Enhancements

- **Secure Aggregation**: Encrypt weight updates so server can't see individual contributions
- **Advanced Composition**: RDP or zCDP for tighter privacy accounting
- **Tamper-Proof Logging**: Move audit log to blockchain or append-only database
- **Membership Inference Testing**: Empirical privacy validation
- **Key Rotation**: Automatic secret key rotation policy
