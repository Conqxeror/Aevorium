import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.schema import get_input_dim, CONTINUOUS_COLUMNS
from common.model import TabularDiffusionModel, DiffusionManager

def main():
    input_dim = get_input_dim()
    model = TabularDiffusionModel(input_dim=input_dim)
    manager = DiffusionManager(model)
    # random sample
    x = manager.sample(1000, input_dim=input_dim)
    arr = x.cpu().numpy()
    import numpy as np
    n_cont = len(CONTINUOUS_COLUMNS)
    cont = arr[:, :n_cont]
    print('cont shape:', cont.shape)
    print('min:', cont.min(), 'max:', cont.max(), 'mean:', cont.mean(), 'std:', cont.std())

if __name__ == '__main__':
    main()
