import os
from common.config import MODEL_DIR

def test_preprocessor_saved_by_client():
    # The client code saves a preprocessor to MODEL_DIR/preprocessor.joblib during initialization
    # Instantiate a client class to trigger saving (import only)
    from node.client import DiffusionClient
    c = DiffusionClient(cid='test-save')
    # Check canonical preprocessor exists
    import glob
    pre_paths = glob.glob(os.path.join(MODEL_DIR, 'preprocessor*.joblib'))
    assert len(pre_paths) > 0, f"Expected at least one preprocessor file in {MODEL_DIR}, but none were found"
    # Clean up created preprocessor files for test hygiene
    for p in pre_paths:
        try:
            os.remove(p)
        except Exception:
            pass
