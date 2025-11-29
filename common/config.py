import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Storage directory (can be overridden by env var for Docker)
STORAGE_DIR = os.getenv("STORAGE_DIR", BASE_DIR)

# File paths
AUDIT_LOG_FILE = os.path.join(STORAGE_DIR, "audit_log.json")
SECRET_KEY_FILE = os.path.join(STORAGE_DIR, "secret.key")
MODEL_DIR = STORAGE_DIR # Where models are saved
SYNTHETIC_DATA_FILE = os.path.join(STORAGE_DIR, "synthetic_data.csv")

# Ensure storage dir exists
os.makedirs(STORAGE_DIR, exist_ok=True)
