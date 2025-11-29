# Shared Schema Definition
# In a real FL system, this would be agreed upon by the consortium.
# Version 2.0: Expanded schema with rich features from Synthea data

# Continuous features: Clinical + Healthcare utilization metrics
CONTINUOUS_COLUMNS = [
    # Clinical metrics
    'Age',              # Patient age (18-90)
    'BMI',              # Body Mass Index (15-50)
    'BloodPressure',    # Systolic BP (80-200)
    'Glucose',          # Blood glucose (50-400)
    # Healthcare utilization metrics
    'EncounterCount',   # Total healthcare encounters
    'MedicationCount',  # Number of medications prescribed
    'ConditionCount',   # Number of diagnosed conditions
    'TotalCost',        # Total healthcare costs ($)
    'ProcedureCount',   # Number of procedures performed
]

CATEGORICAL_COLUMNS = ['Gender', 'Diagnosis', 'EncounterType', 'HasAllergies', 'RiskLevel']

# Pre-defined categories to ensure consistent One-Hot Encoding across clients
# If a client sees a category not in this list, it will be ignored (handle_unknown='ignore')
CATEGORIES = {
    'Gender': ['Male', 'Female', 'Other'],
    'Diagnosis': ['Diabetes', 'Hypertension', 'Healthy', 'HeartDisease', 'ChronicKidneyDisease', 'Cancer'],
    'EncounterType': ['wellness', 'ambulatory', 'outpatient', 'urgentcare', 'emergency', 'inpatient'],
    'HasAllergies': ['Yes', 'No'],
    'RiskLevel': ['Low', 'Medium', 'High']  # Computed from utilization patterns
}

# Feature ranges for clipping/validation
FEATURE_RANGES = {
    'Age': (18, 90),
    'BMI': (15, 50),
    'BloodPressure': (80, 200),
    'Glucose': (50, 400),
    'EncounterCount': (0, 500),
    'MedicationCount': (0, 100),
    'ConditionCount': (0, 30),
    'TotalCost': (0, 300000),
    'ProcedureCount': (0, 200),
}

def get_input_dim():
    """
    Calculates the total input dimension after One-Hot Encoding.
    """
    n_cont = len(CONTINUOUS_COLUMNS)
    n_cat = sum(len(cats) for cats in CATEGORIES.values())
    return n_cont + n_cat


def get_feature_info():
    """
    Returns detailed information about features for documentation/UI.
    """
    return {
        'continuous': {col: FEATURE_RANGES.get(col, (None, None)) for col in CONTINUOUS_COLUMNS},
        'categorical': CATEGORIES,
        'total_dim': get_input_dim()
    }
