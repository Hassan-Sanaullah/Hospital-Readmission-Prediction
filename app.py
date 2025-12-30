import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocess import load_and_preprocess_data, prepare_features

# Load dataset for reference and feature names
df = load_and_preprocess_data("data/diabetic_data.csv")
X_sample, _ = prepare_features(df)  # make sure this uses drop_first=False
feature_names = X_sample.columns

st.title("Hospital Readmission Prediction")
st.markdown(
    "Predict if a patient is likely to be readmitted within 30 days based on their hospital record."
)

# Sidebar: select model
model_choice = st.sidebar.selectbox(
    "Choose Model", ["Logistic_Regression", "Decision_Tree", "Random_Forest"]
)
model = joblib.load(f"models/{model_choice}.pkl")

st.sidebar.header("Patient Information Input")

# Numeric fields
numeric_features = {
    'age': "Patient age (numeric 1-10 mapped from [0-10) - [90-100))",
    'time_in_hospital': "Number of days patient stayed in hospital",
    'num_procedures': "Total number of procedures during stay",
    'num_medications': "Total number of medications prescribed",
    'number_outpatient': "Number of outpatient visits",
    'number_emergency': "Number of emergency visits",
    'number_inpatient': "Number of prior inpatient visits",
    'number_diagnoses': "Total number of diagnoses recorded"
}

inputs = {}
for col, desc in numeric_features.items():
    min_val = int(df[col].min())
    max_val = int(df[col].max())
    default = int(df[col].median())
    inputs[col] = st.sidebar.slider(
        col.replace("_", " ").title(),
        min_value=min_val,
        max_value=max_val,
        value=default,
        help=desc
    )

# Categorical fields
categorical_options = {
    'gender': ('Patient gender', ['Female', 'Male']),
    'race': ('Patient race', df['race'].unique().tolist()),
    'admission_type': ('How the patient was admitted', ['Emergency', 'Elective', 'Newborn', 'Other']),
    'discharge_disposition': ('Where patient went after discharge', ['Home', 'Transfer', 'Other']),
    'admission_source': ('Source of patient admission', ['Physician Referral', 'Clinic', 'Other'])
}

for cat, (desc, options) in categorical_options.items():
    choice = st.sidebar.selectbox(cat.replace("_"," ").title(), options, help=desc)
    for option in options:  # no drop_first
        col_name = f"{cat}_{option}"
        inputs[col_name] = 1 if choice == option else 0

# Medications
med_columns = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'glipizide', 'glyburide', 'insulin'
]
st.sidebar.header("Medications")
for med in med_columns:
    inputs[f"{med}_Yes"] = 1 if st.sidebar.checkbox(med.title(), help=f"Check if patient is taking {med}") else 0

# Change and diabetesMed
inputs['change'] = 1 if st.sidebar.selectbox(
    "Medication Change?", ["No", "Yes"], help="Indicates if patient's diabetes medications were changed during stay"
)=="Yes" else 0

inputs['diabetesMed'] = 1 if st.sidebar.selectbox(
    "On Diabetes Medication?", ["No", "Yes"], help="Indicates if patient is prescribed diabetes medication"
)=="Yes" else 0

# Convert to dataframe
input_df = pd.DataFrame([inputs])

# Add missing columns
for c in X_sample.columns:
    if c not in input_df.columns:
        input_df[c] = 0
input_df = input_df[X_sample.columns]

# Predict button
if st.button("Predict Readmission"):
    # Use predict_proba for all models
    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(input_df)[0][1]  # probability of readmission
        threshold = 0.5
        pred_class = 1 if pred_prob >= threshold else 0
    else:
        # fallback
        pred_class = model.predict(input_df)[0]
        pred_prob = None

    st.subheader("Prediction Result")
    if pred_prob is not None:
        st.write(f"Probability of readmission: **{pred_prob:.2%}**")
    
    if pred_class == 1:
        st.error("⚠️ Patient likely to be readmitted within 30 days")
    else:
        st.success("✅ Patient unlikely to be readmitted within 30 days")
