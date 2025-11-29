import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("Testing Root...")
    resp = requests.get(f"{BASE_URL}/")
    print(resp.json())

def test_generate():
    print("\nTesting Generation...")
    payload = {"n_samples": 100, "output_file": "api_test_data.csv"}
    resp = requests.post(f"{BASE_URL}/generate", json=payload)
    print(resp.json())

def test_audit_log():
    print("\nTesting Audit Log...")
    resp = requests.get(f"{BASE_URL}/audit-log")
    logs = resp.json()
    print(f"Found {len(logs)} log entries.")
    if logs:
        print("Latest log:", logs[-1])

def main():
    try:
        test_root()
        test_generate()
        test_audit_log()
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
