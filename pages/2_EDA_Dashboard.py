import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# --- Page Title ---
st.title("Exploratory Data Analysis (EDA)")

st.markdown("""
### Overview

This section presents a structured exploratory analysis of the air quality dataset.  
The analysis focuses on identifying key temporal trends, pollutant distributions, inter-variable relationships, and station-type variations.  
All visualisations are followed by concise academic interpretations to support data-driven understanding.
""")

# --- Load Cleaned Dataset ---
@st.cache_data
def load_cleaned_data():
    df = pd.read_csv("data/df_final.csv", parse_dates=["datetime"])
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    return df

df = load_cleaned_data()

# --- Filter by station_type ---
st.sidebar.header("Filters")
selected_station_types = st.sidebar.multiselect(
    "Select station types to display:",
    options=df["station_type"].unique(),
    default=df["station_type"].unique()
)
df = df[df["station_type"].isin(selected_station_types)]

# --- Distribution Plots for Major Pollutants ---
st.subheader("Distribution of Major Pollutants")

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(pollutants):
    sns.histplot(df[col], kde=True, bins=40, color='skyblue', ax=axes[i])
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(f"{col} Concentration")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
st.pyplot(fig)

st.markdown("""
*Interpretation:*  
The distributions reveal that most pollutants, particularly **PM2.5** and **PM10**, are **right-skewed**, indicating occasional high-pollution episodes.  
Gaseous pollutants like **SO₂** and **CO** show lower baseline concentrations with occasional peaks, while **O₃** exhibits a more balanced spread, reflecting its secondary pollutant nature and seasonal variability.
""")


# --- Time Series Plots for Key Pollutants ---
st.subheader("Temporal Trends of Key Pollutants")

key_pollutants = ["PM2.5", "CO", "O3"]

for pollutant in key_pollutants:
    st.markdown(f"**{pollutant} Concentration Over Time**")
    fig = px.line(df, x="datetime", y=pollutant, color="station_type", template="simple_white")
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    *Interpretation:*  
    The temporal pattern of **{pollutant}** shows variability across station types.  
    Notably, **PM2.5** exhibits peaks during winter months, potentially due to residential heating and stagnant air conditions.  
    **O₃** levels tend to rise in summer, influenced by photochemical reactions driven by solar intensity.  
    **CO** shows consistent urban elevation, likely linked to vehicular emissions.
    """)

# --- Distributions and Boxplots ---
st.subheader("Distribution and Variation of Pollutants")

with st.container():
    cols = st.columns(3)
    for idx, pollutant in enumerate(key_pollutants):
        with cols[idx]:
            fig = px.box(df, x="station_type", y=pollutant, color="station_type",
                         title=f"{pollutant} Distribution by Station Type", template="simple_white")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
*Interpretation:*  
These boxplots highlight the spread and outlier characteristics for each pollutant across different station types.  
For instance, **urban and industrial areas** show broader spread in **CO** and **PM2.5**, reflecting higher and more variable emissions.  
Ozone appears more balanced, with **rural zones** occasionally exhibiting elevated levels due to regional transport and atmospheric chemistry.
""")

# --- Correlation Heatmap ---
st.subheader("Correlation Analysis Among Features")

numerics = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "DEWP", "PRES", "WSPM"]
corr_matrix = df[numerics].corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Pollutants and Weather Variables")
st.pyplot(fig)

st.markdown("""
*Interpretation:*  
There is a high positive correlation between **PM2.5** and **PM10**, which is expected due to their similar physical properties.  
Negative correlations between **temperature** and **PM2.5** suggest that colder periods may exacerbate particulate pollution.  
**CO** and **NO₂** show moderate positive correlation, indicating common sources such as traffic and combustion.
""")

# --- Diurnal Pattern (Hourly) ---
st.subheader("Diurnal Pattern of PM2.5")

pm_hourly = df.groupby("hour")["PM2.5"].mean().reset_index()
fig = px.line(pm_hourly, x="hour", y="PM2.5", markers=True, template="simple_white")
fig.update_layout(title="Average PM2.5 by Hour of Day", height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
*Interpretation:*  
The hourly pattern of PM2.5 shows clear peaks during **morning (7–9 AM)** and **evening (6–9 PM)** hours, which correspond to traffic rush periods.  
This trend supports the hypothesis that human activity, particularly transportation, significantly contributes to fine particulate pollution in urban areas.
""")

# --- Seasonal Pattern (Monthly) ---
st.subheader("Monthly Average of O₃")

o3_monthly = df.groupby("month")["O3"].mean().reset_index()
fig = px.line(o3_monthly, x="month", y="O3", markers=True, template="simple_white")
fig.update_layout(title="Monthly Average O₃ Levels", height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
*Interpretation:*  
Ozone concentrations show a clear seasonal pattern, increasing during **summer months**.  
This aligns with increased **sunlight and temperature**, both of which accelerate ozone formation through photochemical processes involving NOₓ and VOCs.
""")

# --- Scatter Plot - NO2 vs O3 ---
st.subheader("Relationship Between NO₂ and O₃")

fig_no2_o3 = px.scatter(
    df,
    x="NO2",
    y="O3",
    color="station_type",
    title="NO₂ vs O₃ Relationship by Station Type",
    opacity=0.6,
    template="plotly_white",
    trendline="ols"  # Adds a simple linear trendline for each station type
)
st.plotly_chart(fig_no2_o3, use_container_width=True)

st.markdown("""
*Interpretation:*  
A moderate negative relationship between **NO₂** and **O₃** is visible, particularly in urban and industrial zones.  
This is consistent with atmospheric chemistry, where NO₂ participates in photochemical reactions that both generate and deplete ozone.  
Lower NO₂ levels in rural areas coincide with higher O₃, suggesting regional transport and less NO scavenging in cleaner air masses.
""")


# --- Final Summary & Navigation ---
st.markdown("---")
st.markdown("""
### Summary

The exploratory analysis reveals:
- Strong temporal and seasonal trends in pollutant concentrations
- Clear station-type variation, especially in PM2.5 and CO levels
- Key relationships between meteorological and pollution variables

These insights will inform the next stage of the project: **predictive modelling**.

""")

if st.button("Proceed to Predictive Modelling >"):
    st.switch_page("pages/3_Model_prediction.py")  
