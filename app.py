import streamlit as st
import pandas as pd
import joblib

from preprocess import load_and_preprocess_data, prepare_features
from sklearn.metrics import accuracy_score, precision_score, recall_score


# --------------------------------------------------
# Utility: Convert real age → dataset age category
# --------------------------------------------------
def map_real_age_to_category(age: int) -> int:
    """
    Converts real age (years) to diabetes dataset age category (1–10)
    """
    if age < 10:
        return 1
    elif age < 20:
        return 2
    elif age < 30:
        return 3
    elif age < 40:
        return 4
    elif age < 50:
        return 5
    elif age < 60:
        return 6
    elif age < 70:
        return 7
    elif age < 80:
        return 8
    elif age < 90:
        return 9
    else:
        return 10


# --------------------------------------------------
# Load data (only for feature reference)
# --------------------------------------------------
df = load_and_preprocess_data("data/diabetic_data.csv")
X_sample, y_sample = prepare_features(df)


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    layout="wide"
)

st.title("🏥 Hospital Readmission Prediction")
st.markdown(
    "Predict whether a patient is likely to be **readmitted within 30 days** "
    "based on hospital and treatment information."
)

st.markdown("---")


# --------------------------------------------------
# Sidebar: Model selection
# --------------------------------------------------
st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Logistic_Regression", "Decision_Tree", "Random_Forest"]
)

model = joblib.load(f"models/{model_choice}.pkl")


# --------------------------------------------------
# Sidebar: Patient Inputs
# --------------------------------------------------
st.sidebar.header("Patient Information")

inputs = {}

# ---------- Age (REAL AGE → converted internally)
real_age = st.sidebar.slider(
    "Patient Age (years)",
    min_value=0,
    max_value=100,
    value=50,
    help="Enter the patient's actual age"
)

inputs["age"] = map_real_age_to_category(real_age)


# ---------- Numeric medical features
numeric_features = {
    "time_in_hospital": "Days stayed in hospital",
    "num_procedures": "Number of procedures performed",
    "num_medications": "Number of medications prescribed",
    "number_outpatient": "Outpatient visits in the past year",
    "number_emergency": "Emergency visits in the past year",
    "number_inpatient": "Previous inpatient visits",
    "number_diagnoses": "Total diagnoses recorded"
}

for col, desc in numeric_features.items():
    inputs[col] = st.sidebar.slider(
        col.replace("_", " ").title(),
        min_value=int(df[col].min()),
        max_value=int(df[col].max()),
        value=int(df[col].median()),
        help=desc
    )


# --------------------------------------------------
# Categorical Features (One-Hot Encoded)
# --------------------------------------------------
st.sidebar.header("Demographics & Admission")

categorical_fields = {
    "gender": ["Female", "Male"],
    "race": df["race"].unique().tolist(),
    "admission_type": ["Emergency", "Elective", "Newborn", "Other"],
    "discharge_disposition": ["Home", "Transfer", "Other"],
    "admission_source": ["Physician Referral", "Clinic", "Other"]
}

for feature, options in categorical_fields.items():
    choice = st.sidebar.selectbox(
        feature.replace("_", " ").title(),
        options
    )

    for option in options:
        col_name = f"{feature}_{option}"
        inputs[col_name] = 1 if choice == option else 0


# --------------------------------------------------
# Medications
# --------------------------------------------------
st.sidebar.header("Medications")

medications = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "insulin"
]

for med in medications:
    inputs[f"{med}_Yes"] = 1 if st.sidebar.checkbox(
        med.title(),
        help=f"Check if patient is taking {med}"
    ) else 0


# --------------------------------------------------
# Medication change flags
# --------------------------------------------------
inputs["change"] = 1 if st.sidebar.selectbox(
    "Medication Changed During Stay?",
    ["No", "Yes"]
) == "Yes" else 0

inputs["diabetesMed"] = 1 if st.sidebar.selectbox(
    "On Diabetes Medication?",
    ["No", "Yes"]
) == "Yes" else 0


# --------------------------------------------------
# Prepare input DataFrame
# --------------------------------------------------
input_df = pd.DataFrame([inputs])

# Align with training features
input_df = input_df.reindex(columns=X_sample.columns, fill_value=0)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Readmission"):
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Probability of Readmission:** {pred_prob:.2%}")

    if pred_class == 1:
        st.error("⚠️ Patient is likely to be readmitted within 30 days")
    else:
        st.success("✅ Patient is unlikely to be readmitted within 30 days")

    with st.expander("Model Evaluation Metrics"):
        y_pred = model.predict(X_sample)

        st.write(f"**Accuracy:** {accuracy_score(y_sample, y_pred):.2f}")
        st.write(f"**Precision:** {precision_score(y_sample, y_pred):.2f}")
        st.write(f"**Recall:** {recall_score(y_sample, y_pred):.2f}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Created by <b>Gulina Sajjad</b>"
    "</div>",
    unsafe_allow_html=True
)
