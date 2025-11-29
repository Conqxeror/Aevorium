"""Test named NPZ save/load functionality"""
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model import TabularDiffusionModel
from common.schema import get_input_dim
from common.config import MODEL_DIR

def test_named_save_load():
    print("Testing named NPZ save/load...")
    
    # Create a model with specific weights
    input_dim = get_input_dim()
    model1 = TabularDiffusionModel(input_dim=input_dim)
    
    # Save with names
    state_dict = model1.state_dict()
    params_dict = {name: param.cpu().numpy() for name, param in state_dict.items()}
    
    test_file = os.path.join(MODEL_DIR, 'test_named_save.npz')
    np.savez(test_file, **params_dict)
    
    print(f"✓ Saved {len(params_dict)} parameters with names")
    
    # Load and verify
    data = np.load(test_file)
    file_keys = list(data.files)
    
    print(f"✓ Loaded {len(file_keys)} parameters")
    print(f"  First key: {file_keys[0]}")
    print(f"  Is named save: {not file_keys[0].startswith('arr_')}")
    
    # Load into new model
    model2 = TabularDiffusionModel(input_dim=input_dim)
    loaded_state = {k: torch.tensor(data[k]) for k in file_keys}
    model2.load_state_dict(loaded_state, strict=True)
    
    print("✓ Successfully loaded into new model")
    
    # Verify weights match
    for name, param1 in model1.state_dict().items():
        param2 = model2.state_dict()[name]
        assert torch.allclose(param1, param2), f"Mismatch in {name}"
    
    print("✓ All weights match!")
    
    # Cleanup (close the file first)
    data.close()
    try:
        os.remove(test_file)
    except Exception as e:
        print(f"Note: Could not remove test file: {e}")

if __name__ == '__main__':
    try:
        test_named_save_load()
        print("\n✓ Named save/load test PASSED")
    except AssertionError as e:
        print(f"\n✗ Named save/load test FAILED: {e}")
        sys.exit(1)
