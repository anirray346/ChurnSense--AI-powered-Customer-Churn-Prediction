import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

model = load_model()

st.title("📁 Batch Churn Prediction")
st.markdown("Upload a CSV file to predict churn for multiple customers at once.")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())
    st.write(f"Total customers: {len(df)}")

    if st.button("🔮 Predict All"):
        probs = model.predict_proba(df)[:, 1]
        df['Churn_Probability'] = probs.round(4)
        df['Will_Churn'] = probs > 0.5
        df['Risk_Level'] = ['HIGH 🔴' if p > 0.7 
                            else 'MEDIUM 🟡' if p > 0.4 
                            else 'LOW 🟢' for p in probs]

        st.subheader("Prediction Results")
        st.dataframe(df[['Churn_Probability', 'Will_Churn', 'Risk_Level']])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Predicted Churners", int(df['Will_Churn'].sum()))
        col3.metric("Churn Rate", f"{df['Will_Churn'].mean()*100:.1f}%")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
