"""
Train a simple classifier to distinguish real vs synthetic data and report accuracy/AUC.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import generate_synthetic_data, load_real_healthcare_data
from common.preprocessing import DataPreprocessor
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS


def load_datasets(n_samples=1000, use_real_data=True):
    real_df = load_real_healthcare_data()
    if real_df is None:
        raise RuntimeError("Real data not found")
    if len(real_df) < n_samples:
        real_sample = real_df.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
    else:
        real_sample = real_df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Use generate_samples.py to produce synthetic data or fallback generator
    try:
        # Try reading the generated file
        synth_df = pd.read_csv('synthetic_data.csv')
        if len(synth_df) < n_samples:
            synth_sample = synth_df.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
        else:
            synth_sample = synth_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    except Exception:
        synth_sample = generate_synthetic_data(n_samples, use_real_data=False)

    return real_sample, synth_sample


def prepare_features(real_df, synth_df):
    pre = DataPreprocessor(continuous_cols=CONTINUOUS_COLUMNS, categorical_cols=CATEGORICAL_COLUMNS)
    combined = pd.concat([real_df, synth_df], ignore_index=True)
    pre.fit(combined)
    X = pre.transform(combined)
    y = np.concatenate([np.ones(len(real_df)), np.zeros(len(synth_df))])
    return X, y


def main():
    real_df, synth_df = load_datasets(n_samples=1000, use_real_data=True)
    X, y = prepare_features(real_df, synth_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Discriminator Accuracy: {acc:.4f}")
    print(f"Discriminator AUC: {auc:.4f}")
    print('Confusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
