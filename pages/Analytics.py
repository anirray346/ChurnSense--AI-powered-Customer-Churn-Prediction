import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
df = load_data()

st.title("📈 Analytics — Data Insights")
st.markdown("Explore patterns and trends in customer churn data.")

st.markdown("---")

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn by Internet Service")
    internet_churn = df.groupby('InternetService')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    internet_churn.columns = ['InternetService', 'Churn Rate %']
    fig = px.bar(internet_churn, x='InternetService', y='Churn Rate %',
                 color='Churn Rate %', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn by Payment Method")
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    payment_churn.columns = ['PaymentMethod', 'Churn Rate %']
    fig2 = px.bar(payment_churn, x='PaymentMethod', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Blues')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tenure Distribution by Churn")
    fig3 = px.histogram(df, x='tenure', color='Churn',
                        barmode='overlay',
                        color_discrete_map={'Yes': '#FF7A3D', 'No': '#3D8EFF'})
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Monthly Charges by Churn")
    fig4 = px.box(df, x='Churn', y='MonthlyCharges',
                  color='Churn',
                  color_discrete_map={'Yes': '#FF7A3D', 'No': '#3D8EFF'})
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Row 3 - Model Performance
st.subheader("📊 Model Performance Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC Score", "0.8457")
col2.metric("F1 Score", "0.5771")
col3.metric("Precision", "0.6726")
col4.metric("Recall", "0.5053")

# SHAP plot
st.subheader("SHAP Feature Importance")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
shap_path = os.path.join(BASE_DIR, 'reports', 'shap_importance.png')
if os.path.exists(shap_path):
    st.image(shap_path, caption='SHAP Feature Importance', use_container_width=True)
else:
    st.warning("SHAP plot not found. Run Session 5 notebook first.")
