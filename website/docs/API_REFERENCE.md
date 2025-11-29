# API Reference

Aevorium exposes a REST API using FastAPI v2.0.0.

**Base URL**: `http://localhost:8000`

**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

## Endpoints

### 1. Root
Returns available endpoints and welcome message.

- **URL**: `/`
- **Method**: `GET`
- **Response**:
    ```json
    {
      "message": "Welcome to Aevorium API",
      "docs": "/docs",
      "endpoints": {
        "train": "POST /train",
        "generate": "POST /generate",
        "audit_log": "GET /audit-log",
        "privacy_budget": "GET /privacy-budget",
        "health": "GET /health"
      }
    }
    ```

### 2. Health Check
Health check endpoint for monitoring/orchestration.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
    ```json
    {
      "status": "healthy",
      "service": "Aevorium API",
      "version": "2.0.0"
    }
    ```

### 3. Trigger Training
Starts a new Federated Learning session in the background.

- **URL**: `/train`
- **Method**: `POST`
- **Body**:
    ```json
    {
      "rounds": 3,
      "num_clients": 2
    }
    ```
- **Response**:
    ```json
    {
      "message": "Training started in background",
      "config": { "rounds": 3, "num_clients": 2 }
    }
    ```

### 4. Generate Synthetic Data
Generates synthetic patient records using the latest trained global model.

- **URL**: `/generate`
- **Method**: `POST`
- **Body**:
    ```json
    {
      "n_samples": 1000,
      "output_file": "synthetic_data.csv"
    }
    ```
- **Response**:
    ```json
    {
      "message": "Data generated successfully",
      "path": "synthetic_data.csv"
    }
    ```
- **Error Response** (500):
    ```json
    {
      "detail": "No global model file found. Run training first."
    }
    ```

### 5. Get Audit Log
Retrieves the full governance audit trail.

- **URL**: `/audit-log`
- **Method**: `GET`
- **Response**:
    ```json
    [
      {
        "id": "uuid...",
        "timestamp": "2025-11-28T10:30:00.000000",
        "event_type": "MODEL_SAVED",
        "details": "Saved and encrypted global model round 3..."
      },
      {
        "id": "uuid...",
        "timestamp": "2025-11-28T10:35:00.000000",
        "event_type": "privacy_budget_update",
        "details": { "client_id": "0", "round_num": 3, "epsilon_spent": 5.2 }
      }
    ]
    ```

### 6. Get Privacy Budget
Returns the current privacy budget status including cumulative epsilon and remaining budget.

- **URL**: `/privacy-budget`
- **Method**: `GET`
- **Response**:
    ```json
    {
      "cumulative_epsilon": 15.6,
      "delta": 1e-5,
      "total_budget": 50.0,
      "budget_remaining": 34.4,
      "budget_exhausted_pct": 31.2,
      "num_rounds": 3,
      "avg_epsilon_per_round": 5.2,
      "round_history": [
        {
          "timestamp": "2025-11-28T10:30:00.000000",
          "client_id": "0",
          "round_num": 1,
          "epsilon": 5.2,
          "cumulative_epsilon": 5.2,
          "noise_multiplier": 0.3,
          "epochs": 20
        }
      ]
    }
    ```

### 7. Set Privacy Budget Limit
Sets a total privacy budget limit. Training will fail if budget is exceeded.

- **URL**: `/privacy-budget/set-limit`
- **Method**: `POST`
- **Body**:
    ```json
    {
      "total_budget": 50.0,
      "delta": 1e-5
    }
    ```
- **Response**:
    ```json
    {
      "message": "Privacy budget limit set",
      "total_budget": 50.0,
      "current_epsilon": 15.6,
      "remaining": 34.4
    }
    ```

### 8. Reset Privacy Budget
Resets the privacy budget tracker. **WARNING**: This clears all historical privacy expenditure records.

- **URL**: `/privacy-budget/reset`
- **Method**: `POST`
- **Response**:
    ```json
    {
      "message": "Privacy budget reset",
      "previous_epsilon": 15.6,
      "current_epsilon": 0.0
    }
    ```

## Event Types in Audit Log

| Event Type | Description |
|------------|-------------|
| `TRAINING_STARTED` | Training session initiated |
| `TRAINING_COMPLETED` | Training finished successfully |
| `TRAINING_FAILED` | Training encountered an error |
| `MODEL_SAVED` | Model checkpoint saved and encrypted |
| `ROUND_METRICS` | Per-round aggregated metrics logged |
| `DATA_GENERATION` | Synthetic data generated |
| `privacy_budget_update` | Privacy expenditure recorded |
| `privacy_budget_exceeded` | Privacy budget limit exceeded |
| `privacy_budget_reset` | Privacy tracker reset |
| `PRIVACY_BUDGET_LIMIT_SET` | New privacy budget limit configured |

## Error Handling

All endpoints return standard HTTP error codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request body
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error (details in response)
