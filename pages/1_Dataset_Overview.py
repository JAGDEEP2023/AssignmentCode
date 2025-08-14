import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Dataset Overview")
st.markdown("### Multi-Site Air Quality Data – Beijing")

# --- Introduction ---
st.markdown("""
This project utilises the [**Beijing Multi-Site Air Quality Dataset**](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), which captures air quality and meteorological data from **March 2013 to February 2017**.

Hourly records were collected across **12 monitoring stations** representing diverse settings: urban, suburban, rural, and industrial.  
Each entry includes:
- **Pollutants**: PM2.5, PM10, SO₂, NO₂, CO, and O₃  
- **Weather**: Temperature, pressure, dew point, rainfall, wind speed/direction  

This app analyzes a curated subset from **four representative stations**, chosen for spatial and functional diversity.
""")

# --- Load Datasets ---
@st.cache_data
def load_all_sites():
    urban = pd.read_csv("data/urban.csv")
    suburban = pd.read_csv("data/sub-urban.csv")
    rural = pd.read_csv("data/rural.csv")
    industrial = pd.read_csv("data/industrial.csv")
    return urban, suburban, rural, industrial

urban_df, suburban_df, rural_df, industrial_df = load_all_sites()

# --- Display Each Site in a Separate Container ---
def show_site(title, location_type, description, df, image_url=None):
    with st.container():
        st.markdown(f"#### {title} — *{location_type} Monitoring Station*")
        cols = st.columns([1.2, 5]) if image_url else [None, st]
        
        if image_url:
            with cols[0]:
                st.image(image_url, width=180)
            with cols[1]:
                st.markdown(description)
        else:
            st.markdown(description)
        
        st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("---")

# --- Summary table ---
st.markdown("### Data At a Glance")
summary = pd.DataFrame({
    "Station": ["Guanyuan", "Shunyi", "Huairou", "Dongsi"],
    "Type": ["Urban", "Suburban", "Rural", "Industrial"],
    "Rows": [len(urban_df), len(suburban_df), len(rural_df), len(industrial_df)],
    "Columns": [urban_df.shape[1], suburban_df.shape[1], rural_df.shape[1], industrial_df.shape[1]],
    "% Missing": [
        urban_df.isnull().mean().mean()*100,
        suburban_df.isnull().mean().mean()*100,
        rural_df.isnull().mean().mean()*100,
        industrial_df.isnull().mean().mean()*100
    ]
})
st.dataframe(summary)


# --- Urban ---
show_site(
    "Guanyuan",
    "Urban",
    "Located in central Beijing, Guanyuan represents air quality in densely populated residential and commercial zones, with substantial vehicular and industrial influence.",
    urban_df,
    image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/ChinaGuangyuanZhaohua.png/420px-ChinaGuangyuanZhaohua.png"  
)

# --- Suburban ---
show_site(
    "Shunyi",
    "Suburban",
    "Situated at the northeastern edge of the city, Shunyi reflects transitional air quality between Beijing's core and its periphery — moderately influenced by traffic and development.",
    suburban_df,
    image_url="https://upload.wikimedia.org/wikipedia/commons/7/77/Nancai_Town%2C_Shunyi_District%2C_Beijing_%2820221016%29.png"
)

# --- Rural ---
show_site(
    "Huairou",
    "Rural",
    "This site lies in a mountainous, green area with minimal industrial interference. Huairou serves as a rural baseline to contrast with urban and industrial sites.",
    rural_df,
    image_url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSUO7iyKdWTgALqBIBnmSlpAfJ8AIO_fW20Cg&s"
)

# --- Industrial ---
show_site(
    "Dongsi",
    "Industrial",
    "Located near concentrated manufacturing zones and traffic routes, this station provides critical insights into high-exposure industrial environments.",
    industrial_df,
    image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/China_Beijing_adm_location_map.svg/250px-China_Beijing_adm_location_map.svg.png"
)

# --- Preprocessing Summary
st.markdown("### Pre-Processing Steps Applied")
st.markdown("""
- **Station-wise missing value handling** – Numerical features imputed with median; categorical features filled with mode for each station type.  
- **Duplicate removal** – Eliminated repeated records to ensure data integrity.  
- **Outlier treatment** – Applied Z-score method to detect and remove extreme pollutant values.  
- **Feature engineering** – Extracted `datetime` from year, month, day, and hour; created `month` and other temporal variables for seasonal analysis.  
- **Data type conversion** – Ensured correct formats for datetime, numeric, and categorical fields.  
- **Dataset merging** – Combined urban, suburban, rural, and industrial station datasets into a single unified structure.  
""")
st.markdown("---")

# --- Summary Section ---
st.markdown("### Summary")
st.markdown("""
The dataset’s spatial diversity allows for meaningful comparisons across urban, suburban, rural, and industrial zones.  
By examining each environment individually and collectively, the analysis captures the **full complexity of pollution dynamics** in Beijing.

These four representative stations form the backbone of the next stages: **Exploratory Data Analysis** and **Predictive Modeling**.
""")

# --- Navigation Button ---
st.markdown(" ")
st.markdown("---")
st.markdown(" ")
if st.button("Proceed to Exploratory Data Analysis >"):
    st.switch_page("pages/2_EDA_Dashboard.py")