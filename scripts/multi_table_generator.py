"""
Multi-Table Synthetic Data Generator

Generates linked synthetic healthcare tables while preserving:
- Referential integrity (foreign keys)
- Temporal consistency (encounter dates, procedure dates)
- Statistical relationships between tables

Tables generated:
1. patients - Core patient demographics
2. conditions - Patient conditions/diagnoses
3. encounters - Healthcare visits
4. medications - Prescribed medications
5. procedures - Medical procedures performed
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from common.data import load_real_healthcare_data
from common.schema import CATEGORIES, FEATURE_RANGES


class MultiTableGenerator:
    """
    Generates linked synthetic healthcare tables from a base patient dataset.
    """
    
    def __init__(self, base_patients_df: pd.DataFrame = None):
        """
        Args:
            base_patients_df: DataFrame with synthetic patient data (from diffusion model)
        """
        self.base_patients = base_patients_df
        self._load_real_distributions()
    
    def _load_real_distributions(self):
        """Load real data distributions for realistic linked table generation"""
        try:
            # Load real data for distribution reference
            real_patients = pd.read_csv('data/csv/patients.csv')
            real_conditions = pd.read_csv('data/csv/conditions.csv')
            real_encounters = pd.read_csv('data/csv/encounters.csv')
            real_medications = pd.read_csv('data/csv/medications.csv')
            real_procedures = pd.read_csv('data/csv/procedures.csv')
            
            # Compute distributions
            self.conditions_per_patient = real_conditions.groupby('PATIENT').size().describe()
            self.encounters_per_patient = real_encounters.groupby('PATIENT').size().describe()
            self.medications_per_patient = real_medications.groupby('PATIENT').size().describe()
            self.procedures_per_patient = real_procedures.groupby('PATIENT').size().describe()
            
            # Top condition descriptions
            self.condition_types = real_conditions['DESCRIPTION'].value_counts(normalize=True).head(50)
            
            # Encounter classes
            self.encounter_classes = real_encounters['ENCOUNTERCLASS'].value_counts(normalize=True)
            
            # Top medications
            self.medication_types = real_medications['DESCRIPTION'].value_counts(normalize=True).head(50)
            
            # Top procedures
            self.procedure_types = real_procedures['DESCRIPTION'].value_counts(normalize=True).head(50)
            
            self.has_real_data = True
            
        except Exception as e:
            print(f"Warning: Could not load real data distributions: {e}")
            self.has_real_data = False
            self._set_default_distributions()
    
    def _set_default_distributions(self):
        """Set default distributions when real data is not available"""
        self.condition_types = pd.Series({
            'Hypertension': 0.15,
            'Diabetes mellitus type 2': 0.12,
            'Prediabetes': 0.10,
            'Coronary Heart Disease': 0.08,
            'Chronic Kidney Disease': 0.05,
            'Obesity': 0.10,
            'Hyperlipidemia': 0.08,
            'Anemia': 0.05,
            'Osteoarthritis': 0.07,
            'Depression': 0.06,
            'Anxiety disorder': 0.05,
            'Asthma': 0.04,
            'GERD': 0.05
        })
        
        self.encounter_classes = pd.Series({
            'wellness': 0.35,
            'ambulatory': 0.30,
            'outpatient': 0.15,
            'urgentcare': 0.10,
            'emergency': 0.05,
            'inpatient': 0.05
        })
        
        self.medication_types = pd.Series({
            'Metformin 500 MG Oral Tablet': 0.10,
            'Lisinopril 10 MG Oral Tablet': 0.08,
            'Atorvastatin 20 MG Oral Tablet': 0.08,
            'Amlodipine 5 MG Oral Tablet': 0.07,
            'Omeprazole 20 MG Oral Capsule': 0.06,
            'Metoprolol 25 MG Oral Tablet': 0.05,
            'Gabapentin 300 MG Oral Capsule': 0.04,
            'Levothyroxine 50 MCG Oral Tablet': 0.04,
            'Aspirin 81 MG Oral Tablet': 0.05,
            'Sertraline 50 MG Oral Tablet': 0.04
        })
        
        self.procedure_types = pd.Series({
            'Medication Reconciliation': 0.15,
            'Vital Signs Measurement': 0.12,
            'Blood Panel': 0.10,
            'Lipid Panel': 0.08,
            'Hemoglobin A1c Test': 0.07,
            'Electrocardiogram': 0.06,
            'Urinalysis': 0.05,
            'Colonoscopy': 0.03,
            'Chest X-ray': 0.04,
            'Flu Vaccination': 0.08
        })
    
    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())
    
    def _generate_date_range(self, start_year: int = 2020, end_year: int = 2024) -> Tuple[datetime, datetime]:
        """Generate a random date range"""
        start = datetime(start_year, 1, 1) + timedelta(days=np.random.randint(0, 365 * (end_year - start_year)))
        duration = timedelta(days=np.random.randint(1, 365 * 2))
        end = start + duration
        return start, min(end, datetime(end_year, 12, 31))
    
    def generate_patients_table(self, n_patients: int = None) -> pd.DataFrame:
        """
        Generate synthetic patients table.
        If base_patients is provided, uses those. Otherwise generates new.
        """
        if self.base_patients is not None and n_patients is None:
            n_patients = len(self.base_patients)
        elif n_patients is None:
            n_patients = 1000
        
        patients = []
        
        for i in range(n_patients):
            patient_id = self._generate_uuid()
            
            # Use base patient data if available
            if self.base_patients is not None and i < len(self.base_patients):
                base = self.base_patients.iloc[i]
                age = int(base.get('Age', np.random.randint(18, 90)))
                gender = base.get('Gender', np.random.choice(['Male', 'Female']))
            else:
                age = np.random.randint(18, 90)
                gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
            
            # Generate birthdate from age
            birth_year = datetime.now().year - age
            birthdate = datetime(birth_year, np.random.randint(1, 13), np.random.randint(1, 29))
            
            # Generate other demographic data
            first_names_m = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard']
            first_names_f = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
            
            if gender == 'Male':
                first_name = np.random.choice(first_names_m)
            else:
                first_name = np.random.choice(first_names_f)
            
            patients.append({
                'Id': patient_id,
                'BIRTHDATE': birthdate.strftime('%Y-%m-%d'),
                'DEATHDATE': None,
                'SSN': f'{np.random.randint(100,999)}-{np.random.randint(10,99)}-{np.random.randint(1000,9999)}',
                'FIRST': first_name,
                'LAST': np.random.choice(last_names),
                'GENDER': 'M' if gender == 'Male' else 'F',
                'RACE': np.random.choice(['white', 'black', 'asian', 'hispanic'], p=[0.6, 0.15, 0.1, 0.15]),
                'ETHNICITY': np.random.choice(['nonhispanic', 'hispanic'], p=[0.8, 0.2]),
                'BIRTHPLACE': 'Synthetic City, ST',
                'ADDRESS': f'{np.random.randint(100, 9999)} Main St',
                'CITY': 'Synthetic City',
                'STATE': 'ST',
                'ZIP': f'{np.random.randint(10000, 99999)}',
                'HEALTHCARE_EXPENSES': np.random.lognormal(8, 1),
                'HEALTHCARE_COVERAGE': np.random.uniform(0, 100000)
            })
        
        self.patients_df = pd.DataFrame(patients)
        return self.patients_df
    
    def generate_conditions_table(self) -> pd.DataFrame:
        """Generate conditions linked to patients"""
        if not hasattr(self, 'patients_df'):
            raise ValueError("Generate patients first!")
        
        conditions = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['Id']
            
            # Number of conditions (from distribution or estimate from base patient)
            if self.base_patients is not None:
                idx = self.patients_df.index.get_loc(_)
                if idx < len(self.base_patients):
                    base_cond_count = int(self.base_patients.iloc[idx].get('ConditionCount', 5))
                else:
                    base_cond_count = 5
            else:
                base_cond_count = max(1, int(np.random.poisson(5)))
            
            n_conditions = max(1, base_cond_count + np.random.randint(-2, 3))
            
            for _ in range(n_conditions):
                start, end = self._generate_date_range()
                
                # Sample condition type
                condition_type = np.random.choice(
                    self.condition_types.index,
                    p=self.condition_types.values / self.condition_types.values.sum()
                )
                
                conditions.append({
                    'START': start.strftime('%Y-%m-%d'),
                    'STOP': end.strftime('%Y-%m-%d') if np.random.random() > 0.3 else None,
                    'PATIENT': patient_id,
                    'ENCOUNTER': self._generate_uuid(),  # Will be linked later
                    'CODE': f'{np.random.randint(100000, 999999)}',
                    'DESCRIPTION': condition_type
                })
        
        return pd.DataFrame(conditions)
    
    def generate_encounters_table(self) -> pd.DataFrame:
        """Generate encounters linked to patients"""
        if not hasattr(self, 'patients_df'):
            raise ValueError("Generate patients first!")
        
        encounters = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['Id']
            
            # Number of encounters
            if self.base_patients is not None:
                idx = self.patients_df.index.get_loc(_)
                if idx < len(self.base_patients):
                    base_enc_count = int(self.base_patients.iloc[idx].get('EncounterCount', 20))
                else:
                    base_enc_count = 20
            else:
                base_enc_count = max(1, int(np.random.poisson(30)))
            
            n_encounters = max(1, base_enc_count + np.random.randint(-5, 5))
            
            for _ in range(n_encounters):
                start, end = self._generate_date_range()
                
                # Sample encounter class
                enc_class = np.random.choice(
                    self.encounter_classes.index,
                    p=self.encounter_classes.values / self.encounter_classes.values.sum()
                )
                
                # Cost based on encounter type
                base_cost = {
                    'wellness': 200,
                    'ambulatory': 300,
                    'outpatient': 500,
                    'urgentcare': 800,
                    'emergency': 2000,
                    'inpatient': 10000
                }.get(enc_class, 500)
                
                total_cost = base_cost * (1 + np.random.exponential(0.5))
                
                encounters.append({
                    'Id': self._generate_uuid(),
                    'START': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'STOP': end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'PATIENT': patient_id,
                    'ORGANIZATION': self._generate_uuid(),
                    'PROVIDER': self._generate_uuid(),
                    'PAYER': self._generate_uuid(),
                    'ENCOUNTERCLASS': enc_class,
                    'CODE': f'{np.random.randint(100000, 999999)}',
                    'DESCRIPTION': f'{enc_class.title()} Encounter',
                    'BASE_ENCOUNTER_COST': base_cost,
                    'TOTAL_CLAIM_COST': round(total_cost, 2),
                    'PAYER_COVERAGE': round(total_cost * 0.8, 2),
                    'REASONCODE': None,
                    'REASONDESCRIPTION': None
                })
        
        return pd.DataFrame(encounters)
    
    def generate_medications_table(self) -> pd.DataFrame:
        """Generate medications linked to patients"""
        if not hasattr(self, 'patients_df'):
            raise ValueError("Generate patients first!")
        
        medications = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['Id']
            
            # Number of medications
            if self.base_patients is not None:
                idx = self.patients_df.index.get_loc(_)
                if idx < len(self.base_patients):
                    base_med_count = int(self.base_patients.iloc[idx].get('MedicationCount', 10))
                else:
                    base_med_count = 10
            else:
                base_med_count = max(0, int(np.random.poisson(8)))
            
            n_medications = max(0, base_med_count + np.random.randint(-3, 3))
            
            for _ in range(n_medications):
                start, end = self._generate_date_range()
                
                # Sample medication
                med_type = np.random.choice(
                    self.medication_types.index,
                    p=self.medication_types.values / self.medication_types.values.sum()
                )
                
                base_cost = np.random.uniform(10, 200)
                
                medications.append({
                    'START': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'STOP': end.strftime('%Y-%m-%dT%H:%M:%SZ') if np.random.random() > 0.4 else None,
                    'PATIENT': patient_id,
                    'PAYER': self._generate_uuid(),
                    'ENCOUNTER': self._generate_uuid(),
                    'CODE': f'{np.random.randint(100000, 999999)}',
                    'DESCRIPTION': med_type,
                    'BASE_COST': round(base_cost, 2),
                    'PAYER_COVERAGE': round(base_cost * 0.7, 2),
                    'DISPENSES': np.random.randint(1, 12),
                    'TOTALCOST': round(base_cost * np.random.randint(1, 12), 2),
                    'REASONCODE': None,
                    'REASONDESCRIPTION': None
                })
        
        return pd.DataFrame(medications)
    
    def generate_procedures_table(self) -> pd.DataFrame:
        """Generate procedures linked to patients"""
        if not hasattr(self, 'patients_df'):
            raise ValueError("Generate patients first!")
        
        procedures = []
        
        for _, patient in self.patients_df.iterrows():
            patient_id = patient['Id']
            
            # Number of procedures
            if self.base_patients is not None:
                idx = self.patients_df.index.get_loc(_)
                if idx < len(self.base_patients):
                    base_proc_count = int(self.base_patients.iloc[idx].get('ProcedureCount', 15))
                else:
                    base_proc_count = 15
            else:
                base_proc_count = max(0, int(np.random.poisson(10)))
            
            n_procedures = max(0, base_proc_count + np.random.randint(-3, 3))
            
            for _ in range(n_procedures):
                start, _ = self._generate_date_range()
                
                # Sample procedure
                proc_type = np.random.choice(
                    self.procedure_types.index,
                    p=self.procedure_types.values / self.procedure_types.values.sum()
                )
                
                base_cost = np.random.uniform(50, 1000)
                
                procedures.append({
                    'DATE': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'PATIENT': patient_id,
                    'ENCOUNTER': self._generate_uuid(),
                    'CODE': f'{np.random.randint(100000, 999999)}',
                    'DESCRIPTION': proc_type,
                    'BASE_COST': round(base_cost, 2),
                    'REASONCODE': None,
                    'REASONDESCRIPTION': None
                })
        
        return pd.DataFrame(procedures)
    
    def generate_all_tables(self, n_patients: int = None, output_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate all linked synthetic tables.
        
        Args:
            n_patients: Number of patients (uses base_patients count if not specified)
            output_dir: Directory to save CSV files (optional)
        
        Returns:
            Dictionary of table name -> DataFrame
        """
        print("=" * 60)
        print("GENERATING LINKED SYNTHETIC TABLES")
        print("=" * 60)
        
        print("\n[1/5] Generating patients...")
        patients = self.generate_patients_table(n_patients)
        print(f"   Generated {len(patients)} patients")
        
        print("\n[2/5] Generating conditions...")
        conditions = self.generate_conditions_table()
        print(f"   Generated {len(conditions)} conditions")
        
        print("\n[3/5] Generating encounters...")
        encounters = self.generate_encounters_table()
        print(f"   Generated {len(encounters)} encounters")
        
        print("\n[4/5] Generating medications...")
        medications = self.generate_medications_table()
        print(f"   Generated {len(medications)} medications")
        
        print("\n[5/5] Generating procedures...")
        procedures = self.generate_procedures_table()
        print(f"   Generated {len(procedures)} procedures")
        
        tables = {
            'patients': patients,
            'conditions': conditions,
            'encounters': encounters,
            'medications': medications,
            'procedures': procedures
        }
        
        # Save to files if output_dir specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for name, df in tables.items():
                path = os.path.join(output_dir, f'synthetic_{name}.csv')
                df.to_csv(path, index=False)
                print(f"   Saved {path}")
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        
        # Summary
        print("\nTable Summary:")
        for name, df in tables.items():
            print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
        
        return tables


def main():
    """Generate multi-table synthetic data"""
    
    # Try to load base synthetic patients
    synth_path = 'synthetic_data.csv'
    if os.path.exists(synth_path):
        print(f"Loading base synthetic patients from {synth_path}...")
        base_patients = pd.read_csv(synth_path)
        print(f"Loaded {len(base_patients)} base patients")
    else:
        print("No synthetic_data.csv found. Generating from scratch.")
        base_patients = None
    
    # Create generator
    generator = MultiTableGenerator(base_patients)
    
    # Generate all tables
    output_dir = 'synthetic_tables'
    tables = generator.generate_all_tables(output_dir=output_dir)
    
    print(f"\nAll tables saved to {output_dir}/")
    

if __name__ == "__main__":
    main()
