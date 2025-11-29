import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from node.client import DiffusionClient
from common.config import MODEL_DIR

c = DiffusionClient(cid='debug')
print('MODEL_DIR:', MODEL_DIR)
print('Files in MODEL_DIR:')
for f in os.listdir(MODEL_DIR):
    if f.startswith('preprocessor'):
        print(f)

# print if specific file exists
print('canonical exists?', os.path.exists(os.path.join(MODEL_DIR, 'preprocessor.joblib')))
