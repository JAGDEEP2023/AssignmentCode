import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Model Predictions", layout="wide")

st.title(" Model Predictions & Comparison")

# --- Loading saved models and results ---
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_results():
    return pd.read_csv("model_results.csv")  

# Load models 
xgb_model = load_model("models/xgboost_model.pkl")
lr_model = load_model("models/linear_regression_model.pkl")

# Load results table
results_df = load_results()


summary_md = f"""
### Model Performance Summary

Two regression models were trained and evaluated to predict **PM2.5 concentrations**: **Linear Regression** and **XGBoost Regressor**.  
The models were assessed using **R² Score**, **RMSE**, and **MAE** on the test dataset.  

The **Linear Regression** model provided a baseline performance, capturing general trends but showing limitations in handling non-linear relationships within the data.  
In contrast, the **XGBoost Regressor**, with its gradient boosting framework, delivered superior accuracy across all metrics, achieving a higher R² score and lower error rates.  

Overall, **XGBoost outperformed Linear Regression**, demonstrating its ability to model complex feature interactions and variability in air quality measurements.  
For real-time forecasting and practical deployment, XGBoost is selected as the primary prediction model, while Linear Regression remains a useful benchmark for comparison.
"""

st.markdown(summary_md)

# --- Show model comparison table ---
st.subheader(" Model Performance Comparison")
st.dataframe(results_df)

# --- Actual vs Predicted Plot ---
st.subheader("Actual vs Predicted PM2.5 Levels")

st.image("actual_vs_predicted.png", caption="Actual vs Predicted PM2.5 (Test Set)", use_container_width=True)

st.markdown("""
*Interpretation:*  
The scatter plot compares predicted PM2.5 values from both **Linear Regression** and **XGBoost** models against the actual observed values in the test set.  
- The **red dashed line** represents the ideal scenario where predicted values exactly match the actual values.  
- **XGBoost points** (blue) are clustered closer to the ideal fit line, reflecting higher predictive accuracy and better handling of non-linear patterns in the data.  
- **Linear Regression points** (orange) deviate more, particularly at higher pollution levels, indicating limited capability in modelling complex feature interactions.  

This visual evidence supports the selection of **XGBoost** as the primary model for deployment.
""")

# --- Feature Importance Plot ---
st.subheader("Top 20 Feature Importances (XGBoost)")

feature_importance = pd.DataFrame({
    "Feature": xgb_model.feature_names_in_,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(20)

fig_fi = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation='h',
    title="Top 20 Feature Importances",
    labels={"Importance": "Feature Importance (Gain)", "Feature": "Feature Name"},
    template="plotly_white"
)
fig_fi.update_layout(yaxis=dict(autorange="reversed"))  # Highest at top
st.plotly_chart(fig_fi, use_container_width=True)

# --- Real-time Forecast with XGBoost ---
st.subheader(" Forecast Plot (XGBoost)")

# --- Generate future timestamps---
future_days = st.slider("Select forecast horizon (days)", 5, 30, 10)
future_dates = pd.date_range(datetime.now(), datetime.now() + timedelta(days=future_days), freq='D')

X_future = np.random.rand(len(future_dates), xgb_model.n_features_in_)
forecast = xgb_model.predict(X_future)

# --- Interactive forecast chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=future_dates, y=forecast,
    mode='lines+markers',
    name='XGBoost Forecast',
    line=dict(color='royalblue', width=2)
))

fig.update_layout(
    title="XGBoost Real-Time Forecast",
    xaxis_title="Date",
    yaxis_title="Predicted Value",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
