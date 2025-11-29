import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
from common.schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, CATEGORIES, FEATURE_RANGES

# Path to real Synthea CSV data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'csv')

# Cache for loaded real data
_REAL_DATA_CACHE = None


def load_real_healthcare_data():
    """
    Load and process ENRICHED healthcare data from Synthea CSV files.
    
    Extracts features from multiple tables:
    - patients.csv: Age, Gender
    - observations.csv: BMI, BloodPressure, Glucose
    - conditions.csv: Diagnosis, ConditionCount
    - encounters.csv: EncounterCount, EncounterType, TotalCost
    - medications.csv: MedicationCount
    - procedures.csv: ProcedureCount
    - allergies.csv: HasAllergies
    
    Returns DataFrame with all CONTINUOUS_COLUMNS and CATEGORICAL_COLUMNS from schema.
    """
    global _REAL_DATA_CACHE
    if _REAL_DATA_CACHE is not None:
        return _REAL_DATA_CACHE.copy()
    
    # File paths
    files = {
        'patients': os.path.join(DATA_DIR, 'patients.csv'),
        'observations': os.path.join(DATA_DIR, 'observations.csv'),
        'conditions': os.path.join(DATA_DIR, 'conditions.csv'),
        'encounters': os.path.join(DATA_DIR, 'encounters.csv'),
        'medications': os.path.join(DATA_DIR, 'medications.csv'),
        'procedures': os.path.join(DATA_DIR, 'procedures.csv'),
        'allergies': os.path.join(DATA_DIR, 'allergies.csv'),
    }
    
    required_files = ['patients', 'observations', 'conditions']
    if not all(os.path.exists(files[f]) for f in required_files):
        print("Warning: Required data files not found, falling back to synthetic data")
        return None
    
    print("=" * 60)
    print("Loading ENRICHED healthcare data from Synthea CSVs...")
    print("=" * 60)
    
    # =====================================================================
    # 1. LOAD PATIENTS - Base table with Age and Gender
    # =====================================================================
    print("\n[1/7] Loading patients...")
    patients = pd.read_csv(files['patients'], usecols=['Id', 'BIRTHDATE', 'GENDER'])
    patients = patients.rename(columns={'Id': 'PATIENT'})
    
    reference_year = 2024
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    patients['Age'] = reference_year - patients['BIRTHDATE'].dt.year
    patients = patients[(patients['Age'] >= 18) & (patients['Age'] <= 90)]
    
    gender_map = {'M': 'Male', 'F': 'Female'}
    patients['Gender'] = patients['GENDER'].map(gender_map).fillna('Other')
    patients = patients[['PATIENT', 'Age', 'Gender']]
    print(f"   Loaded {len(patients)} adult patients")
    
    # =====================================================================
    # 2. LOAD OBSERVATIONS - Clinical metrics (BMI, BP, Glucose)
    # =====================================================================
    print("\n[2/7] Loading clinical observations...")
    obs_codes = {
        '39156-5': 'BMI',
        '8480-6': 'BloodPressure',
        '2339-0': 'Glucose',
        '2345-7': 'Glucose',  # Alternative glucose code
    }
    
    observations_list = []
    chunksize = 50000
    for chunk in pd.read_csv(files['observations'], usecols=['PATIENT', 'CODE', 'VALUE'], chunksize=chunksize):
        chunk = chunk[chunk['CODE'].isin(obs_codes.keys())]
        if not chunk.empty:
            observations_list.append(chunk)
    
    if observations_list:
        observations = pd.concat(observations_list, ignore_index=True)
        observations['VALUE'] = pd.to_numeric(observations['VALUE'], errors='coerce')
        observations['Metric'] = observations['CODE'].map(obs_codes)
        obs_pivot = observations.groupby(['PATIENT', 'Metric'])['VALUE'].mean().unstack().reset_index()
    else:
        obs_pivot = pd.DataFrame(columns=['PATIENT', 'BMI', 'BloodPressure', 'Glucose'])
    
    df = patients.merge(obs_pivot, on='PATIENT', how='left')
    print(f"   Merged clinical observations")
    
    # =====================================================================
    # 3. LOAD CONDITIONS - Diagnosis and ConditionCount
    # =====================================================================
    print("\n[3/7] Loading conditions...")
    conditions = pd.read_csv(files['conditions'], usecols=['PATIENT', 'DESCRIPTION'])
    
    # Count conditions per patient
    condition_counts = conditions.groupby('PATIENT').size().reset_index(name='ConditionCount')
    
    # Map conditions to diagnosis categories
    def map_diagnosis(desc):
        desc_lower = str(desc).lower()
        if 'diabetes' in desc_lower or 'prediabetes' in desc_lower:
            return 'Diabetes'
        elif 'hypertension' in desc_lower or 'blood pressure' in desc_lower:
            return 'Hypertension'
        elif 'heart' in desc_lower or 'cardiac' in desc_lower or 'coronary' in desc_lower:
            return 'HeartDisease'
        elif 'kidney' in desc_lower or 'renal' in desc_lower:
            return 'ChronicKidneyDisease'
        elif 'cancer' in desc_lower or 'neoplasm' in desc_lower or 'malignant' in desc_lower or 'carcinoma' in desc_lower:
            return 'Cancer'
        return None
    
    conditions['DiagCategory'] = conditions['DESCRIPTION'].apply(map_diagnosis)
    conditions_filtered = conditions[conditions['DiagCategory'].notna()]
    
    # Get primary diagnosis per patient (priority order)
    priority = {'Cancer': 5, 'ChronicKidneyDisease': 4, 'Diabetes': 3, 'HeartDisease': 2, 'Hypertension': 1}
    conditions_filtered = conditions_filtered.copy()
    conditions_filtered['Priority'] = conditions_filtered['DiagCategory'].map(priority)
    patient_diag = conditions_filtered.groupby('PATIENT').apply(
        lambda x: x.loc[x['Priority'].idxmax(), 'DiagCategory'] if len(x) > 0 else 'Healthy',
        include_groups=False
    ).reset_index()
    patient_diag.columns = ['PATIENT', 'Diagnosis']
    
    df = df.merge(condition_counts, on='PATIENT', how='left')
    df = df.merge(patient_diag, on='PATIENT', how='left')
    df['Diagnosis'] = df['Diagnosis'].fillna('Healthy')
    df['ConditionCount'] = df['ConditionCount'].fillna(0)
    print(f"   Processed {len(conditions)} condition records")
    
    # =====================================================================
    # 4. LOAD ENCOUNTERS - EncounterCount, EncounterType, TotalCost
    # =====================================================================
    print("\n[4/7] Loading encounters...")
    if os.path.exists(files['encounters']):
        encounters = pd.read_csv(files['encounters'], usecols=['PATIENT', 'ENCOUNTERCLASS', 'TOTAL_CLAIM_COST'])
        
        # Encounter count per patient
        enc_count = encounters.groupby('PATIENT').size().reset_index(name='EncounterCount')
        
        # Most frequent encounter type per patient
        enc_type = encounters.groupby('PATIENT')['ENCOUNTERCLASS'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'wellness'
        ).reset_index()
        enc_type.columns = ['PATIENT', 'EncounterType']
        
        # Total healthcare cost per patient
        enc_cost = encounters.groupby('PATIENT')['TOTAL_CLAIM_COST'].sum().reset_index()
        enc_cost.columns = ['PATIENT', 'TotalCost']
        
        df = df.merge(enc_count, on='PATIENT', how='left')
        df = df.merge(enc_type, on='PATIENT', how='left')
        df = df.merge(enc_cost, on='PATIENT', how='left')
        print(f"   Processed {len(encounters)} encounter records")
    else:
        df['EncounterCount'] = 10
        df['EncounterType'] = 'wellness'
        df['TotalCost'] = 5000
    
    # =====================================================================
    # 5. LOAD MEDICATIONS - MedicationCount
    # =====================================================================
    print("\n[5/7] Loading medications...")
    if os.path.exists(files['medications']):
        medications = pd.read_csv(files['medications'], usecols=['PATIENT'])
        med_count = medications.groupby('PATIENT').size().reset_index(name='MedicationCount')
        df = df.merge(med_count, on='PATIENT', how='left')
        print(f"   Processed {len(medications)} medication records")
    else:
        df['MedicationCount'] = 5
    
    # =====================================================================
    # 6. LOAD PROCEDURES - ProcedureCount
    # =====================================================================
    print("\n[6/7] Loading procedures...")
    if os.path.exists(files['procedures']):
        procedures = pd.read_csv(files['procedures'], usecols=['PATIENT'])
        proc_count = procedures.groupby('PATIENT').size().reset_index(name='ProcedureCount')
        df = df.merge(proc_count, on='PATIENT', how='left')
        print(f"   Processed {len(procedures)} procedure records")
    else:
        df['ProcedureCount'] = 5
    
    # =====================================================================
    # 7. LOAD ALLERGIES - HasAllergies
    # =====================================================================
    print("\n[7/7] Loading allergies...")
    if os.path.exists(files['allergies']):
        allergies = pd.read_csv(files['allergies'], usecols=['PATIENT'])
        allergy_patients = set(allergies['PATIENT'].unique())
        df['HasAllergies'] = df['PATIENT'].apply(lambda x: 'Yes' if x in allergy_patients else 'No')
        print(f"   Processed {len(allergies)} allergy records")
    else:
        df['HasAllergies'] = 'No'
    
    # =====================================================================
    # COMPUTE RISK LEVEL - Derived feature based on utilization patterns
    # =====================================================================
    print("\n[Derived] Computing RiskLevel...")
    def compute_risk_level(row):
        score = 0
        # High encounter count
        if row.get('EncounterCount', 0) > 50:
            score += 2
        elif row.get('EncounterCount', 0) > 20:
            score += 1
        # High medication count
        if row.get('MedicationCount', 0) > 20:
            score += 2
        elif row.get('MedicationCount', 0) > 10:
            score += 1
        # High condition count
        if row.get('ConditionCount', 0) > 10:
            score += 2
        elif row.get('ConditionCount', 0) > 5:
            score += 1
        # High cost
        if row.get('TotalCost', 0) > 50000:
            score += 2
        elif row.get('TotalCost', 0) > 10000:
            score += 1
        
        if score >= 5:
            return 'High'
        elif score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    df['RiskLevel'] = df.apply(compute_risk_level, axis=1)
    
    # =====================================================================
    # FINAL CLEANUP - Fill missing values, clip to ranges
    # =====================================================================
    print("\n[Final] Cleaning and validating data...")
    
    # Fill missing continuous values with medians
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            median_val = df[col].median() if df[col].notna().any() else 0
            df[col] = df[col].fillna(median_val)
            # Clip to valid ranges
            if col in FEATURE_RANGES:
                min_val, max_val = FEATURE_RANGES[col]
                df[col] = df[col].clip(min_val, max_val)
    
    # Fill missing categorical values with defaults
    default_cats = {
        'Gender': 'Other',
        'Diagnosis': 'Healthy',
        'EncounterType': 'wellness',
        'HasAllergies': 'No',
        'RiskLevel': 'Low'
    }
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(default_cats.get(col, CATEGORIES[col][0]))
            # Ensure values are in valid categories
            valid_cats = set(CATEGORIES.get(col, []))
            if valid_cats:
                df[col] = df[col].apply(lambda x: x if x in valid_cats else default_cats.get(col, list(valid_cats)[0]))
    
    # Select final columns matching schema
    final_cols = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
    df = df[[c for c in final_cols if c in df.columns]].copy()
    
    # Drop any remaining NaN rows
    df = df.dropna()
    
    # =====================================================================
    # SUMMARY STATISTICS
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"LOADED {len(df)} ENRICHED PATIENT RECORDS")
    print("=" * 60)
    print("\nContinuous Features:")
    for col in CONTINUOUS_COLUMNS:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, range=[{df[col].min():.0f}, {df[col].max():.0f}]")
    
    print("\nCategorical Features:")
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            dist = df[col].value_counts(normalize=True).head(4).to_dict()
            dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in dist.items()])
            print(f"  {col}: {dist_str}")
    
    _REAL_DATA_CACHE = df
    return df.copy()


def generate_synthetic_data(n_samples=1000, use_real_data=True):
    """
    Returns healthcare data for training.
    If use_real_data=True and real data is available, samples from real data.
    Otherwise, generates synthetic data with realistic distributions.
    
    Returns DataFrame with all columns from CONTINUOUS_COLUMNS and CATEGORICAL_COLUMNS.
    """
    if use_real_data:
        real_df = load_real_healthcare_data()
        if real_df is not None and len(real_df) > 0:
            # Sample with replacement if we need more than available
            if n_samples <= len(real_df):
                return real_df.sample(n=n_samples, random_state=None).reset_index(drop=True)
            else:
                # Bootstrap sample
                return real_df.sample(n=n_samples, replace=True, random_state=None).reset_index(drop=True)
    
    # Fallback: generate synthetic data with correlated features
    print("Using synthetic data generator (fallback)...")
    np.random.seed(None)  # Use random seed for variety
    
    # --- Continuous Clinical Features ---
    # Age: Normal distribution around 50
    age = np.random.normal(50, 15, n_samples)
    age = np.clip(age, 18, 90)
    
    # BMI: Correlated with Age slightly
    bmi = np.random.normal(27, 5, n_samples) + (age - 50) * 0.1
    bmi = np.clip(bmi, 15, 50)
    
    # Blood Pressure: Correlated with Age and BMI
    bp = np.random.normal(125, 15, n_samples) + (age - 50) * 0.5 + (bmi - 25) * 1.0
    bp = np.clip(bp, 80, 200)
    
    # Glucose: Correlated with BMI
    glucose = np.random.normal(105, 25, n_samples) + (bmi - 25) * 2
    glucose = np.clip(glucose, 50, 400)
    
    # --- Healthcare Utilization Features (correlated with clinical severity) ---
    # Encounter count: Higher for older, higher BMI patients
    severity_score = (age - 18) / 72 + (bmi - 15) / 35 + (glucose - 50) / 350
    encounter_count = np.random.poisson(20 + severity_score * 30, n_samples)
    encounter_count = np.clip(encounter_count, 1, 500)
    
    # Medication count: Correlated with severity
    medication_count = np.random.poisson(5 + severity_score * 15, n_samples)
    medication_count = np.clip(medication_count, 0, 100)
    
    # Condition count: Correlated with age and severity
    condition_count = np.random.poisson(3 + (age - 18) / 20 + severity_score * 5, n_samples)
    condition_count = np.clip(condition_count, 0, 30)
    
    # Total cost: Log-normal, correlated with encounters and severity
    log_cost = np.random.normal(8 + severity_score, 1, n_samples)  # Log-scale
    total_cost = np.exp(log_cost) + encounter_count * 50
    total_cost = np.clip(total_cost, 100, 300000)
    
    # Procedure count: Correlated with conditions
    procedure_count = np.random.poisson(condition_count * 2 + 3, n_samples)
    procedure_count = np.clip(procedure_count, 0, 200)
    
    # --- Categorical Features ---
    # Gender: 48% Male, 48% Female, 4% Other
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
    
    # Diagnosis: Correlated with clinical metrics
    diagnosis = []
    for g, b, a in zip(glucose, bmi, age):
        if g > 140 or b > 35:
            diagnosis.append(np.random.choice(['Diabetes', 'HeartDisease', 'ChronicKidneyDisease'], p=[0.5, 0.3, 0.2]))
        elif g > 120:
            diagnosis.append(np.random.choice(['Hypertension', 'HeartDisease'], p=[0.7, 0.3]))
        elif a > 65:
            diagnosis.append(np.random.choice(['Hypertension', 'Healthy', 'Cancer'], p=[0.4, 0.4, 0.2]))
        else:
            diagnosis.append('Healthy')
    
    # Encounter type: Correlated with severity
    encounter_type = []
    for sev in severity_score:
        if sev > 0.7:
            encounter_type.append(np.random.choice(['inpatient', 'emergency', 'urgentcare'], p=[0.4, 0.3, 0.3]))
        elif sev > 0.4:
            encounter_type.append(np.random.choice(['ambulatory', 'outpatient', 'urgentcare'], p=[0.5, 0.35, 0.15]))
        else:
            encounter_type.append(np.random.choice(['wellness', 'ambulatory'], p=[0.6, 0.4]))
    
    # Has allergies: ~20% have allergies
    has_allergies = np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
    
    # Risk level: Computed based on utilization
    risk_level = []
    for enc, med, cond, cost in zip(encounter_count, medication_count, condition_count, total_cost):
        score = 0
        if enc > 50: score += 2
        elif enc > 20: score += 1
        if med > 20: score += 2
        elif med > 10: score += 1
        if cond > 10: score += 2
        elif cond > 5: score += 1
        if cost > 50000: score += 2
        elif cost > 10000: score += 1
        
        if score >= 5:
            risk_level.append('High')
        elif score >= 2:
            risk_level.append('Medium')
        else:
            risk_level.append('Low')
            
    data = pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'BloodPressure': bp,
        'Glucose': glucose,
        'EncounterCount': encounter_count,
        'MedicationCount': medication_count,
        'ConditionCount': condition_count,
        'TotalCost': total_cost,
        'ProcedureCount': procedure_count,
        'Gender': gender,
        'Diagnosis': diagnosis,
        'EncounterType': encounter_type,
        'HasAllergies': has_allergies,
        'RiskLevel': risk_level
    })
    
    return data

class TabularDataset(Dataset):
    def __init__(self, data):
        # data should be a pre-processed numpy array or tensor
        if isinstance(data, pd.DataFrame):
            raise ValueError("TabularDataset expects a numpy array or tensor, not a DataFrame. Use DataPreprocessor first.")
        self.data = data.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

def get_dataloader(data_array, batch_size=32):
    dataset = TabularDataset(data_array)
    # Use pin_memory for GPU if available, and allow configuration of num_workers via env var
    # Use a slightly higher default for num_workers to improve throughput on multi-core machines
    default_workers = max(2, int(os.getenv('DATALOADER_NUM_WORKERS', '4')))
    num_workers = int(os.getenv('DATALOADER_NUM_WORKERS', str(default_workers)))
    pin_memory = True if torch.cuda.is_available() else False
    # Use persistent_workers for better performance with multi-threaded loaders (PyTorch >=1.7)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
