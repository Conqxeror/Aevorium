import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import generate_synthetic_data
from common.preprocessing import DataPreprocessor
from common.config import MODEL_DIR
import os

ref = generate_synthetic_data(20)
pre = DataPreprocessor(continuous_cols=['Age','BMI','BloodPressure','Glucose'], categorical_cols=['Gender','Diagnosis'])
pre.fit(ref)
path = os.path.join(MODEL_DIR,'preprocessor_test.joblib')
print('Saving to', path)
pre.save(path)
print('Exists after save?', os.path.exists(path))
