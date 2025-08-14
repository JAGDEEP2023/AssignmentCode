import streamlit as st

# Page configuration
st.set_page_config(page_title="Air Quality Analysis Tool", layout="wide")

# Page Header
st.title("Air Quality Analysis Tool")
st.subheader("A Comprehensive Dashboard for Monitoring, Exploration, and Prediction")
st.markdown("---")

# Hero Image
st.image(
    "https://cdn.pixabay.com/photo/2016/10/19/12/38/industry-1752876_640.png",
    caption="Air pollution near industrial zones",
    use_container_width=True
)

st.markdown("## The Problem")
left1, right1 = st.columns([2, 1])
with left1:
    st.markdown("""
Urbanization and industrial growth have significantly impacted air quality, especially in major cities like Beijing. Concentrations of pollutants such as PM2.5, NO₂, and CO have exceeded safe levels, posing threats to public health and environmental sustainability. Identifying spatial pollution patterns is essential for developing data-driven mitigation strategies.
    """)
with right1:
    st.image(
        "https://www.mdpi.com/sustainability/sustainability-10-03228/article_deploy/html/images/sustainability-10-03228-g001.png",
        caption="Understanding urban pollution patterns",
        use_container_width=True
    )

st.markdown("## Project Objective")
st.markdown("""
This application analyzes multi-site air quality data from urban, suburban, rural, and industrial monitoring stations across Beijing. It compares pollutant distributions across regions and applies predictive modeling (Linear Regression, XGBoost) to forecast PM2.5 levels.
""")


st.markdown("## Dataset Source")
left2, right2 = st.columns([1.5, 2])
with left2:
    st.image(
        "https://royalsocietypublishing.org/cms/asset/a44412dd-a970-4ef3-a768-435922ba4367/rspa20170457f01.jpg",
        caption="Multi-site monitoring",
        use_container_width=True
    )
with right2:
    st.markdown("""
This project utilizes the **Beijing Multi-Site Air Quality Dataset**, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).  
It includes time-series data from multiple air quality monitoring stations located across Beijing, specifically: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong, and Xizhimen.
These stations capture detailed measurements of key pollutants including PM2.5, PM10, NO₂, CO, SO₂, and O₃, making the dataset suitable for analyzing temporal and spatial air quality patterns across urban and suburban areas.
    """)

st.markdown("## Why This Matters")
st.markdown("""
Air pollution is a leading contributor to respiratory and cardiovascular diseases and is linked to millions of premature deaths annually. Gaining granular insights into pollution sources and temporal patterns enables policymakers to design targeted and effective interventions.
""")

st.markdown("## Methodology Overview")
st.markdown("""
Raw datasets from multiple station types were merged into a unified structure. Exploratory Data Analysis (EDA) was used to extract visual insights into pollution variability. Two regression models (Linear Regression and XGBoost) were implemented for PM2.5 prediction and feature importance analysis.
""")

st.markdown("## Application Guide")
guide1, guide2 = st.columns(2)
with guide1:
    st.markdown("""
- **Dataset Overview**  
  Review station-level data sources, dimensions, and pollutant coverage.
  
- **Exploratory Analysis**  
  Interactive charts showing trends, anomalies, and correlations.
    """)
with guide2:
    st.markdown("""
- **Modeling**  
  Learn how the models perform, compare them, and explore feature importance.

- **User Navigation**  
  Use the sidebar or buttons below to access each section of the dashboard.
    """)


st.markdown("---")

# Navigation Buttons 
cta1, cta2, cta3 = st.columns(3)
with cta1:
    st.page_link("pages/1_Dataset_Overview.py", label="Go to Dataset Overview >")
with cta2:
    st.page_link("pages/2_EDA_Dashboard.py", label="Go to EDA Dashboard >")
with cta3:
    st.page_link("pages/3_Model_prediction.py", label="Go to Modeling Page >")
