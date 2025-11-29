import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import load_real_healthcare_data

real = load_real_healthcare_data()
synth = pd.read_csv('synthetic_data.csv')

print('--- Categorical JS Divergence')
for col in ['Gender','Diagnosis']:
    r = real[col].value_counts(normalize=True)
    s = synth[col].value_counts(normalize=True)
    idx = sorted(set(r.index) | set(s.index))
    rvals = np.array([r.get(i,0) for i in idx])
    svals = np.array([s.get(i,0) for i in idx])
    js = jensenshannon(rvals, svals, base=2)
    print(f"{col}: JS divergence (bits) = {js:.4f}")

print('\n--- Continuous Mean/Std differences')
for col in ['Age','BMI','BloodPressure','Glucose']:
    r_mean, r_std = real[col].mean(), real[col].std()
    s_mean, s_std = synth[col].mean(), synth[col].std()
    print(f"{col}: real mean={r_mean:.2f}, synth mean={s_mean:.2f}, mean diff={abs(r_mean-s_mean):.2f}, std diff={abs(r_std-s_std):.2f}")
