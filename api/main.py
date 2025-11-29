from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import subprocess
import sys
import os
import json
import io
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from generate_samples import generate_synthetic_dataset
from common.governance import get_logs, log_event, get_privacy_tracker
from common.config import SYNTHETIC_DATA_FILE

app = FastAPI(
    title="Aevorium API", 
    description="Federated Synthetic Data Platform for Healthcare",
    version="2.0.0"
)

class GenerateRequest(BaseModel):
    n_samples: int = 1000
    output_file: str = SYNTHETIC_DATA_FILE

class TrainingRequest(BaseModel):
    rounds: int = 3
    num_clients: int = 2

class PrivacyBudgetRequest(BaseModel):
    total_budget: float = None
    delta: float = 1e-5

def run_training_simulation(rounds: int):
    # This is a simplified runner that triggers the PowerShell script or similar logic
    # In a real app, this would orchestrate K8s jobs.
    # Here we will try to run the run_poc.ps1 script
    
    log_event("TRAINING_STARTED", f"Started training for {rounds} rounds")
    
    try:
        # We use the powershell script as it handles multiple processes
        # Note: This might be blocking or tricky to capture output in real-time
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "run_poc.ps1"], check=True)
        log_event("TRAINING_COMPLETED", "Training finished successfully")
    except Exception as e:
        log_event("TRAINING_FAILED", str(e))
        print(f"Training failed: {e}")

@app.post("/train")
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Triggers a new federated learning training session.
    """
    background_tasks.add_task(run_training_simulation, request.rounds)
    return {"message": "Training started in background", "config": request}

@app.post("/generate")
async def generate_data(request: GenerateRequest):
    """
    Generates synthetic data using the latest trained model.
    """
    try:
        output_path = generate_synthetic_dataset(request.n_samples, request.output_file)
        log_event("DATA_GENERATION", {
            "n_samples": request.n_samples,
            "output_file": request.output_file,
            "path": output_path
        })
        return {"message": "Data generated successfully", "path": output_path, "n_samples": request.n_samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit-log")
async def read_audit_log():
    """
    Retrieves the governance audit log.
    """
    return get_logs()


def sanitize_for_json(obj):
    """Replace inf/nan with JSON-compatible values"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return None if obj > 0 else None
        elif math.isnan(obj):
            return None
        return obj
    return obj


@app.get("/privacy-budget")
async def get_privacy_budget():
    """
    Get the current privacy budget status.
    
    Returns:
    - cumulative_epsilon: Total privacy spent so far
    - budget_remaining: Remaining budget (if limit set)
    - round_history: Last 5 training rounds with privacy info
    """
    try:
        tracker = get_privacy_tracker()
        summary = tracker.get_summary()
        # Sanitize infinity values for JSON
        return sanitize_for_json(summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/privacy-budget/set-limit")
async def set_privacy_budget_limit(request: PrivacyBudgetRequest):
    """
    Set a total privacy budget limit.
    Training will fail if budget is exceeded.
    """
    try:
        tracker = get_privacy_tracker(total_budget=request.total_budget, delta=request.delta)
        tracker.total_budget = request.total_budget
        tracker._save_state()
        
        log_event("PRIVACY_BUDGET_LIMIT_SET", {
            "total_budget": request.total_budget,
            "delta": request.delta
        })
        
        return {
            "message": "Privacy budget limit set",
            "total_budget": request.total_budget,
            "current_epsilon": tracker.cumulative_epsilon,
            "remaining": tracker.get_remaining_budget()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/privacy-budget/reset")
async def reset_privacy_budget():
    """
    Reset the privacy budget tracker.
    WARNING: This clears all historical privacy expenditure records.
    """
    try:
        tracker = get_privacy_tracker()
        old_epsilon = tracker.cumulative_epsilon
        tracker.reset()
        
        return {
            "message": "Privacy budget reset",
            "previous_epsilon": old_epsilon,
            "current_epsilon": 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "service": "Aevorium API",
        "version": "2.0.0"
    }


@app.get("/training-status")
async def get_training_status():
    """
    Get the current training status from audit logs.
    """
    logs = get_logs()
    training_logs = [l for l in logs if l.get('event_type', '').startswith('TRAINING')]
    
    if not training_logs:
        return {"status": "idle", "message": "No training has been run"}
    
    last_log = training_logs[-1]
    event_type = last_log.get('event_type', '')
    
    if event_type == 'TRAINING_STARTED':
        return {"status": "running", "started_at": last_log.get('timestamp')}
    elif event_type == 'TRAINING_COMPLETED':
        return {"status": "completed", "completed_at": last_log.get('timestamp')}
    elif event_type == 'TRAINING_FAILED':
        return {"status": "failed", "error": last_log.get('details'), "failed_at": last_log.get('timestamp')}
    
    return {"status": "unknown"}


@app.get("/")
async def root():
    return {
        "message": "Welcome to Aevorium API",
        "docs": "/docs",
        "endpoints": {
            "train": "POST /train",
            "generate": "POST /generate",
            "dataset": "GET /dataset",
            "dataset_download": "GET /dataset/download",
            "dataset_stats": "GET /dataset/stats",
            "audit_log": "GET /audit-log",
            "privacy_budget": "GET /privacy-budget",
            "health": "GET /health"
        }
    }


# =====================================================================
# DATASET ENDPOINTS
# =====================================================================

@app.get("/dataset")
async def get_dataset(limit: int = 100, offset: int = 0):
    """
    Get the generated synthetic dataset as JSON.
    
    Args:
        limit: Number of rows to return (max 1000)
        offset: Starting row index
    """
    try:
        # Find the most recent synthetic data file
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
        
        df = pd.read_csv(SYNTHETIC_DATA_FILE)
        total_rows = len(df)
        
        # Apply pagination
        limit = min(limit, 1000)  # Cap at 1000 rows
        df_page = df.iloc[offset:offset + limit]
        
        return {
            "total_rows": total_rows,
            "offset": offset,
            "limit": limit,
            "returned_rows": len(df_page),
            "columns": list(df.columns),
            "data": df_page.to_dict(orient="records")
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/download")
async def download_dataset():
    """
    Download the synthetic dataset as a CSV file.
    """
    try:
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
        
        return FileResponse(
            path=SYNTHETIC_DATA_FILE,
            media_type="text/csv",
            filename="synthetic_healthcare_data.csv"
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/stats")
async def get_dataset_stats():
    """
    Get statistics and visualizations for the synthetic dataset.
    
    Returns summary statistics, distributions, and column info.
    """
    try:
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
        
        df = pd.read_csv(SYNTHETIC_DATA_FILE)
        
        stats = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "continuous": {},
            "categorical": {}
        }
        
        # Analyze each column
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Continuous column - convert all numpy types to Python native
                stats["continuous"][col] = {
                    "mean": float(round(df[col].mean(), 2)),
                    "std": float(round(df[col].std(), 2)),
                    "min": float(round(df[col].min(), 2)),
                    "max": float(round(df[col].max(), 2)),
                    "median": float(round(df[col].median(), 2)),
                    "q25": float(round(df[col].quantile(0.25), 2)),
                    "q75": float(round(df[col].quantile(0.75), 2)),
                    # Histogram data (10 bins)
                    "histogram": get_histogram(df[col])
                }
            else:
                # Categorical column
                value_counts = df[col].value_counts()
                stats["categorical"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "distribution": {
                        str(k): {"count": int(v), "percentage": float(round(v / len(df) * 100, 1))}
                        for k, v in value_counts.items()
                    }
                }
        
        return stats
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_histogram(series, bins=10):
    """Generate histogram data for a numeric series"""
    try:
        import numpy as np
        counts, bin_edges = np.histogram(series.dropna(), bins=bins)
        return {
            "counts": [int(c) for c in counts],
            "bin_edges": [round(float(e), 2) for e in bin_edges]
        }
    except Exception:
        return {"counts": [], "bin_edges": []}


@app.get("/dataset/sample")
async def get_dataset_sample(n: int = 10):
    """
    Get a random sample from the dataset.
    """
    try:
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
        
        df = pd.read_csv(SYNTHETIC_DATA_FILE)
        n = min(n, len(df), 100)  # Cap at 100
        sample = df.sample(n=n)
        
        return {
            "sample_size": n,
            "data": sample.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validate")
async def validate_synthetic_data():
    """
    Validate the quality of generated synthetic data by comparing 
    distributions with expected real data patterns.
    
    Returns quality scores for continuous and categorical columns.
    """
    try:
        if not os.path.exists(SYNTHETIC_DATA_FILE):
            raise HTTPException(status_code=404, detail="No synthetic data has been generated yet")
        
        df = pd.read_csv(SYNTHETIC_DATA_FILE)
        
        # Quality scoring
        continuous_scores = []
        categorical_scores = []
        
        # Define expected ranges for continuous columns (based on healthcare data)
        expected_ranges = {
            'Age': (18, 90),
            'BMI': (15, 50),
            'BloodPressure': (80, 200),
            'Glucose': (50, 200),
            'EncounterCount': (1, 500),
            'MedicationCount': (0, 200),
            'ConditionCount': (0, 30),
            'TotalCost': (0, 100000),
            'ProcedureCount': (0, 300),
        }
        
        # Score continuous columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if values are within expected range
                if col in expected_ranges:
                    min_exp, max_exp = expected_ranges[col]
                    in_range = ((df[col] >= min_exp) & (df[col] <= max_exp)).mean()
                    continuous_scores.append(in_range * 100)
                else:
                    # Default score based on non-null and finite values
                    valid = df[col].notna() & (df[col] != float('inf')) & (df[col] != float('-inf'))
                    continuous_scores.append(valid.mean() * 100)
            else:
                # Categorical columns - score based on having valid values
                valid = df[col].notna() & (df[col] != '')
                categorical_scores.append(valid.mean() * 100)
        
        # Calculate overall scores
        continuous_score = sum(continuous_scores) / len(continuous_scores) if continuous_scores else 0
        categorical_score = sum(categorical_scores) / len(categorical_scores) if categorical_scores else 0
        overall_score = (continuous_score + categorical_score) / 2 if (continuous_scores or categorical_scores) else 0
        
        return {
            "overall_score": round(overall_score, 1),
            "continuous_score": round(continuous_score, 1),
            "categorical_score": round(categorical_score, 1),
            "total_rows": len(df),
            "columns_analyzed": len(df.columns),
            "status": "excellent" if overall_score >= 80 else "good" if overall_score >= 60 else "fair"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-history")
async def clear_history():
    """
    Clear all history data including:
    - Audit logs
    - Generated synthetic datasets
    - Privacy budget history (optional reset)
    
    This is a destructive operation and cannot be undone.
    """
    try:
        cleared = {
            "audit_log": False,
            "synthetic_data": False,
            "files_deleted": []
        }
        
        # Clear audit log
        from common.config import AUDIT_LOG_FILE
        if os.path.exists(AUDIT_LOG_FILE):
            with open(AUDIT_LOG_FILE, 'w') as f:
                json.dump([], f)
            cleared["audit_log"] = True
        
        # Delete synthetic data files
        synthetic_files = [
            SYNTHETIC_DATA_FILE,
            "synthetic_data.csv",
            "api_test_data.csv"
        ]
        
        for filepath in synthetic_files:
            full_path = filepath if os.path.isabs(filepath) else os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath)
            if os.path.exists(full_path):
                os.remove(full_path)
                cleared["files_deleted"].append(os.path.basename(full_path))
                cleared["synthetic_data"] = True
        
        # Log this action (creates new log entry)
        log_event("HISTORY_CLEARED", {
            "cleared_audit_log": cleared["audit_log"],
            "files_deleted": cleared["files_deleted"]
        })
        
        return {
            "message": "History cleared successfully",
            "details": cleared
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
