# Aevorium — Hackathon Presentation Notes (📋 Final Markdown)

## Title
Aevorium — Federated Synthetic Data Platform

---

## 1) Full Project Explanation (Simple, Presentation-Ready)
Aevorium helps organizations train machine learning models on private data (like patient health records) without the raw data ever leaving its owner’s network. It enables multiple hospitals or labs to collaboratively train a generative model (a diffusion model) that produces safe, realistic synthetic healthcare datasets. This synthetic data can be used for research, model development, or simulations without exposing sensitive patient records.

Why it matters:
- Preserves privacy while enabling collaboration across organizations.
- Produces realistic synthetic data that preserves statistical properties of the original data.
- Combines federated learning + differential privacy + secure storage for a complete privacy-safe solution.

---

## 2) Complete Workflow Breakdown (Step-by-Step; “beauty” explained)
Explain each stage clearly and why it’s elegant:

1. Setup & Shared Schema (Files: schema.py, preprocessing.py)
   - Purpose: Ensure every participant (client) understands the same data format and feature encodings.
   - Beauty: Schema agreement keeps model inputs consistent; one-hot categories are predefined so no back-and-forth encoding mismatch occurs.

2. Local Synthetic Data Generation for PoC (File: data.py)
   - Purpose: For demos and local PoC, each node generates a realistic dataset with age, BMI, blood pressure, glucose, gender, and diagnosis.
   - Beauty: Data generation produces representative distributions and correlations, letting you test end-to-end without real data.

3. Local Preprocessing (File: preprocessing.py)
   - Purpose: Each node standardizes numeric features and one-hot-encodes categorical features before training.
   - Beauty: The preprocessor can be saved and used to transform/inverse transform data, enabling end-to-end reproducibility.

4. Local Training with Differential Privacy (File: client.py)
   - Purpose: Each client trains a copy of a diffusion model on their local dataset while guaranteeing privacy using Opacus (DP).
   - Beauty: Local training never shares raw data: it shares model gradients/weights only. Opacus ensures these updates preserve privacy.

5. Federated Aggregation (File: server.py)
   - Purpose: The server (Flower) coordinates rounds of training, aggregates weights (e.g., FedAvg), and saves the encrypted global model after each round.
   - Beauty: The strategy is extensible, supports secure aggregation, and saves encrypted model artifacts automatically with an audit trail.

6. Encrypted Model Storage & Governance (Files: security.py, governance.py)
   - Purpose: Models are encrypted using Fernet (cryptography), and audit events (training start/end, model saves, generation events) are logged to audit_log.json.
   - Beauty: Provides tamper-resistant workflow logs and ensures models are safe at rest; helpful for compliance and reproducibility.

7. Synthetic Data Generation from Global Model (File: generate_samples.py)
   - Purpose: After training, the server’s global weights are decrypted, loaded into a local diffusion model, and used to sample synthetic tabular data.
   - Beauty: The powerful diffusion sampler produces high-fidelity samples that preserve statistical relationships while providing privacy.

8. Validation & Comparison (File: validate_data.py)
   - Purpose: Compare real and synthetic datasets on summary stats, correlations, and histograms to verify fidelity.
   - Beauty: Direct, interpretable metrics show how well the synthetic data matches reality — essential for trust in the synthetic outputs.

---

## 3) How to Run the Project (Without Docker) — Beginner-Friendly
These steps assume you’re on Windows with PowerShell (pwsh). Use a Python 3.10+ environment (requirements specify up-to-date packages).

### Environment Setup
1. Open PowerShell (pwsh).
2. Create and activate a virtual environment:
   ```pwsh
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```pwsh
   pip install -r requirements.txt
   ```
4. Optional: verify GPU is used with PyTorch if available.

### Quick Sanity Check
- Run a dry-run to verify imports and basic pipeline:
  ```pwsh
  python dry_run.py
  ```

### Run the federated PoC (manual)
Option 1 — Manual Start:
- Start the server (one terminal):
  ```pwsh
  python server/server.py
  ```
- Start two clients (two separate terminals):
  ```pwsh
  python node/client.py
  ```
  Repeat in another terminal to start more clients if you'd like.

Option 2 — Automated PoC:
- Use included PowerShell script to start server and two clients:
  ```pwsh
  ./run_poc.ps1
  ```
  This script launches server & two clients, waits for the server to finish, then automatically generates samples and runs validation.

### Generate Synthetic Data
- Once training finishes, generate synthetic data from the latest model:
  ```pwsh
  python generate_samples.py
  ```
  Output is saved as synthetic_data.csv by default.

### Run the API (optional)
- Start the FastAPI app (API endpoints for training, generating data, and audit logs):
  ```pwsh
  python api/main.py
  ```
  or, for live reloading:
  ```pwsh
  python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  ```

### Validate Generated Data
- Run:
  ```pwsh
  python validate_data.py
  ```
  This prints statistics and optionally saves comparison plots (if Matplotlib is installed).

---

## 4) How to Use the Project (Examples & Simple API Calls)
Short, practical examples — use these in your demo:

### 1) CLI Demo Steps (Quick)
- Terminal 1: Start server
  ```pwsh
  python server/server.py
  ```
- Terminal 2 & 3: Start clients (one per terminal)
  ```pwsh
  python node/client.py
  ```
- Terminal 4: Generate synthetic data (after server has saved a model)
  ```pwsh
  python generate_samples.py
  ```
- Terminal 5: Validate data
  ```pwsh
  python validate_data.py
  ```

### 2) Demo Using the API
Start the API (see above), then call endpoints from a new terminal:

- Trigger training (Background task):
  ```pwsh
  Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/train" -ContentType "application/json" -Body (@{rounds=3; num_clients=2} | ConvertTo-Json)
  ```
- Generate synthetic data (saves to default file):
  ```pwsh
  Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/generate" -ContentType "application/json" -Body (@{n_samples=1000; output_file='synthetic_data.csv'} | ConvertTo-Json)
  ```
- Get audit log:
  ```pwsh
  Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/audit-log"
  ```
- Or using `curl` (cross-platform):
  ```bash
  curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d '{"n_samples": 100}'
  ```

### 3) Direct CLI for Validation
- `python generate_samples.py` to generate synthetic_data.csv.
- `python validate_data.py` to compare synthetic with expected real distributions.

---

## 5) Key Highlights for Your Presentation (Slide-Ready)
Use these crisp bullets when pitching:

- Problem: Real healthcare (and similar) data is sensitive — sharing it is risky and regulated.
- Solution: Aevorium produces high-fidelity synthetic data via federated learning and differential privacy so institutions can collaborate without exposing raw records.
- Privacy-first approach:
  - Local training at each node (client) — raw data never leaves local site.
  - Opacus differential privacy during training (prevents re-identification).
  - Encrypted model checkpoints (Fernet) for safe storage and retrieval.
  - Audit trail logs every important operation to audit_log.json.
- Generative model: Diffusion-based MLP for tabular data — simple architecture, powerful sampling.
- Clean schema management: schema.py ensures consistent features and categories across clients.
- Standalone PoC: Works locally without cloud or K8s — run_poc.ps1 and dry_run.py enable easy demos.
- Validation pipeline: validate_data.py compares real vs synthetic distributions and correlations (trust is critical).
- Developer-friendly structure: separate server, node, and api modules — easy to extend (e.g., change the model, add secure aggregation).
- Tech stack: PyTorch, Flower (FL), Opacus (DP), FastAPI, with robust encryption and governance.

---

## 6) Talking Points & “Why this is Cool” (Short)
- “We let institutions Pool Knowledge, Not Data.” (One-line description suitable for an elevator pitch).
- Combines three modern privacy measures: federated learning, differential privacy, and encryption — layered protection.
- Demonstrable fidelity (statistical checks) to show utility for downstream tasks.
- Modular design: swap the model architecture, alter privacy budget, or add more clients without major rework.
- Great for hackathon demos: local PoC script and quick “generate-and-validate” pipeline.

---

## Bonus — Quick Troubleshooting & Tips
- If you see “No global model file found”, run training first (run server + clients).
- If `decrypt_file()` fails, check config.py’s `SECRET_KEY_FILE` or regenerate using `common/security.generate_key()`; avoid overwriting keys after model saving.
- If Opacus throws wrapping errors, the code uses a deep copy of the model to prevent double-wrapping — that’s intentional.
- Use `python dry_run.py` to sanity-check preprocessor/modelcompatibility before starting a full run.

---

## Files to reference during your presentation (Show these in slides)
- model.py — diffusion model & manager  
- schema.py — agreed schema and input dimension calculation  
- client.py — local training + Opacus integration  
- server.py — federated server with save/strategy  
- generate_samples.py — sample generation from global model  
- main.py — API endpoints (training/generate/audit-log)  
- run_poc.ps1 & dry_run.py — PoC and sanity checks  
- security.py, governance.py — encryption & audit logging

---

## Final Notes & Demo Recommendation (Short)
- For a live demo: run run_poc.ps1 in PowerShell; show synthetic_data.csv generated and open validate_data.py output to show stat match. Then show the audit log (`GET /audit-log`) and demonstrate the API `/generate` endpoint generating fresh synthetic data on-demand.
- Keep the explanation high-level: “we train together, but securely, and then we produce synthetic data for safe sharing and modeling”.

---
