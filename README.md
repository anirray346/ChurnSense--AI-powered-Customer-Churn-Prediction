# 🏦 ChurnSense — Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-red)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Live Demo
👉 **[Click here to open the app]([https://your-streamlit-url-here](https://churnsense--ai-powered-customer-churn-prediction-app.streamlit.app/))**

---

## 📌 Project Overview
ChurnSense is an end-to-end Machine Learning project that predicts whether a telecom customer will churn (leave the service) based on their usage patterns, contract type, and billing information.

Built as part of the **IoT Academy Data Science & ML Internship**, this project covers the complete ML pipeline — from raw data to a deployed web application.

---

## 📊 Dataset
- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes/No → 1/0)
- **Class Imbalance:** 73.5% No Churn / 26.5% Churn

---

## 🧠 ML Pipeline

### Sessions Completed:
| Session | Topic | Status |
|---------|-------|--------|
| 1 | Project Setup & Data Loading | ✅ Done |
| 2 | Exploratory Data Analysis | ✅ Done |
| 3 | Preprocessing & Feature Engineering | ✅ Done |
| 4 | Model Building & Training | ✅ Done |
| 5 | Model Evaluation & SHAP Explainability | ✅ Done |
| 6 | Production Pipeline & Unit Testing | ✅ Done |
| 7 | Streamlit Web App | ✅ Done |
| 8 | Deployment | ✅ Done |

---

## 🤖 Models Trained & Compared

| Model | AUC | F1 | Precision | Recall |
|-------|-----|----|-----------|--------|
| **Gradient Boosting** ⭐ | **0.8457** | 0.5771 | 0.6726 | 0.5053 |
| Logistic Regression | 0.8413 | 0.6136 | 0.5043 | 0.7834 |
| XGBoost | 0.8254 | 0.5726 | 0.6060 | 0.5428 |
| Random Forest | 0.8208 | 0.5856 | 0.5566 | 0.6176 |
| Decision Tree | 0.6537 | 0.4906 | 0.4946 | 0.4866 |

**Best Model:** Gradient Boosting (after GridSearchCV tuning)
- `learning_rate = 0.05`
- `max_depth = 3`
- `n_estimators = 100`

---

## 🔍 Key Business Insights

1. **Contract type is the #1 churn driver** — Month-to-month customers churn at 42.7% vs only 2.8% for two-year contracts
2. **Long tenure = loyalty** — Customers with high tenure are significantly less likely to churn
3. **Higher monthly charges = higher churn risk** — Positive correlation with churn
4. **No Tech Support or Online Security = higher churn** — Value-added services retain customers
5. **Fiber optic customers churn more** — Possibly due to higher costs or competition

---

## 🚀 Web App Features

### 📊 Dashboard
- Total customers, churn rate, avg monthly charges, model AUC
- Churn distribution pie chart
- Churn rate by contract type bar chart

### 🔮 Predict
- Single customer churn prediction
- Input form with sliders and dropdowns
- Risk level output (LOW 🟢 / MEDIUM 🟡 / HIGH 🔴)

### 📁 Batch Prediction
- Upload CSV file with multiple customers
- Get churn predictions for all customers at once
- Download results as CSV

### 📈 Analytics
- Churn by Internet Service and Payment Method
- Tenure and Monthly Charges distribution
- Model performance metrics
- SHAP Feature Importance, Summary and Waterfall plots

---

## 📁 Project Structure
ChurnSense/
├── app/
│   ├── Dashboard.py          # Main dashboard page
│   └── pages/
│       ├── 01_Predict.py     # Single prediction page
│       ├── 02_Batch.py       # Batch prediction page
│       └── 03_Analytics.py   # Analytics page
├── src/
│   ├── preprocessing.py      # Preprocessing functions
│   ├── model.py              # Model loading and prediction
│   └── utils.py              # Helper functions
├── tests/
│   └── test_pipeline.py      # Pytest unit tests
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── best_model.pkl        # Saved Gradient Boosting model
├── reports/
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── shap_waterfall.png
│   ├── roc_curves.png
│   └── confusion_matrix.png
├── notebooks/
│   └── Customer churn.ipynb
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

```bash
# Clone the repo
git clone https://github.com/anirray346/churnsense.git
cd churnsense

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Dashboard.py
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

Expected output:
tests/test_pipeline.py::test_model_loads                PASSED ✅
tests/test_pipeline.py::test_prediction_is_probability  PASSED ✅
tests/test_pipeline.py::test_prediction_has_risk_level  PASSED ✅

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models & preprocessing |
| XGBoost | Gradient boosting |
| SHAP | Model explainability |
| Matplotlib & Seaborn | EDA visualizations |
| Plotly | Interactive charts |
| Streamlit | Web app framework |
| Joblib | Model serialization |
| Pytest | Unit testing |
| Git & GitHub | Version control |

---

## 👤 Author

**Anirban Ray**
- 💼 [LinkedIn](https://linkedin.com/in/anirban-ray05)
- 🐙 [GitHub](https://github.com/anirray346)
- 🌐 [Streamlit Portfolio](https://share.streamlit.io/user/anirray346)

---

## 🏆 Certification
This project was built as part of the **IoT Academy Data Science & ML Internship Program**.

---

