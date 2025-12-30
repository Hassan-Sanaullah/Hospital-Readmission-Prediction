import pandas as pd
import numpy as np
from sklearn.utils import resample

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Remove invalid data
    df = df[(df[['diag_1', 'diag_2', 'diag_3']] != '?').all(axis=1)]
    df = df[df['race'] != '?']
    df = df[df['gender'] != 'Unknown/Invalid']
    df = df[df['discharge_disposition_id'] != 11]  # Remove expired

    # Encode age to numeric
    age_mapping = {f"[{i*10}-{(i+1)*10})": i+1 for i in range(10)}
    df['age'] = df['age'].replace(age_mapping)

    # Keep first encounter per patient
    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')

    # Drop unnecessary columns
    df = df.drop(['weight','payer_code','medical_specialty','encounter_id','patient_nbr'], axis=1)

    # Map numeric IDs to descriptive words
    df['admission_type'] = df['admission_type_id'].replace({
        1: "Emergency", 2:"Emergency", 3:"Elective", 4:"Newborn",
        5:"Other", 6:"Other", 7:"Other", 8:"Other"
    })
    df['discharge_disposition'] = df['discharge_disposition_id'].replace({
        1:"Home", 2:"Transfer", 10:"Other", 18:"Other"
    })
    df['admission_source'] = df['admission_source_id'].replace({
        1:"Physician Referral", 4:"Clinic", 9:"Other", 11:"Other"
    })
    df = df.drop(['admission_type_id','discharge_disposition_id','admission_source_id'], axis=1)

    # Encode binary columns
    df['change'] = df['change'].map({'Ch':1, 'No':0})
    df['diabetesMed'] = df['diabetesMed'].map({'Yes':1, 'No':0})
    df['gender'] = df['gender'].map({'Male':1, 'Female':0})
    df['readmitted'] = df['readmitted'].map({'>30':0, '<30':1, 'NO':0})

    return df

def prepare_features(df):
    # Split features and target
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']

    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include='object').columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Balance dataset
    df_majority = X[y==0]
    df_minority = X[y==1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    X_bal = pd.concat([df_majority, df_minority_upsampled])
    y_bal = pd.Series([0]*len(df_majority) + [1]*len(df_minority_upsampled))

    return X_bal, y_bal
