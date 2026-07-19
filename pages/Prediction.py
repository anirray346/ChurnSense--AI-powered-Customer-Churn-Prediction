import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

st.set_page_config(page_title="Predict Churn", page_icon="🔮", layout="wide")

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))

model = load_model()

st.title("🔮 Predict Customer Churn")
st.markdown("Fill in the customer details below to predict churn probability.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

if st.button("🔮 Predict"):
    customer = {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': 'Yes',
        'MultipleLines': 'No', 'InternetService': internet,
        'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': tech_support,
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': monthly_charges * tenure
    }

    prob = model.predict_proba(pd.DataFrame([customer]))[0][1]

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Will Churn", "Yes ⚠️" if prob > 0.5 else "No ✅")
    col3.metric("Risk Level", "HIGH 🔴" if prob > 0.7 else "MEDIUM 🟡" if prob > 0.4 else "LOW 🟢")

    if prob > 0.5:
        st.error("⚠️ This customer is at HIGH risk of churning!")
    else:
        st.success("✅ This customer is likely to stay!")