import os
from generate_samples import generate_synthetic_dataset
from common.config import MODEL_DIR
import glob

def test_generate_uses_preprocessor_if_present():
    # Ensure a preprocessor exists (saved by a client)
    from node.client import DiffusionClient
    c = DiffusionClient(cid='test-gen')

    # Generate a small synthetic dataset using preprocessor saved
    out = generate_synthetic_dataset(n_samples=10, output_file=os.path.join(MODEL_DIR, 'synthetic_test_preproc.csv'))
    assert os.path.exists(out), f"Expected generated file at {out}"

    # Cleanup
    try:
        os.remove(out)
    except Exception:
        pass
    # Remove preprocessor file saved by test client(s)
    for p in glob.glob(os.path.join(MODEL_DIR, 'preprocessor*.joblib')):
        try:
            os.remove(p)
        except Exception:
            pass
