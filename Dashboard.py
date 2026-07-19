
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

st.set_page_config(
    page_title="ChurnSense",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))

model = load_model()
df = load_data()

st.title("🏦 ChurnSense — Customer Analytics")
st.markdown("*AI-Powered Churn Prediction Dashboard*")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
churn_rate = (df['Churn'] == 'Yes').mean() * 100
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-1.2%", delta_color="inverse")
col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
col4.metric("Model AUC Score", "0.8457")

st.markdown("---")

# Churn Distribution Chart
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    fig = px.pie(churn_counts, values='Count', names='Churn',
                 color_discrete_sequence=['#3D8EFF', '#FF7A3D'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Churn by Contract Type")
    contract_churn = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    contract_churn.columns = ['Contract', 'Churn Rate %']
    fig2 = px.bar(contract_churn, x='Contract', y='Churn Rate %',
                  color='Churn Rate %', color_continuous_scale='Reds')
    st.plotly_chart(fig2, use_container_width=True)
