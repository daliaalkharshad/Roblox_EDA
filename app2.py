import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="Roblox EDA Dashboard", layout="wide")

# -----------------------
# Data Loading & Cleaning
# -----------------------
@st.cache_data
def load_and_clean_data():
    """
    Loads raw CSV and applies cleaning logic from EDA_Project (4).ipynb:
    - Imputes missing Genres
    - Converts K/M/B strings to numeric
    - Converts dates and creates 'YearsSinceCreation'
    """
    df = pd.read_csv("my_data.csv")
    
    # 1. Fill missing Genre with mode
    df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])
    
    # 2. Conversion Function for K, M, B Suffixes
    def convert(x):
        if isinstance(x, str):
            x = x.replace('+', '').replace(',', '')
            if "K" in x: return float(x.replace("K", "")) * 1000
            elif "M" in x: return float(x.replace("M", "")) * 1000000
            elif "B" in x: return float(x.replace("B", "")) * 1000000000
            return float(x)
        return x

    for col in ['Visits', 'Likes', 'Dislikes', 'Favorites', 'Active']:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    # 3. Date Processing & Feature Engineering
    df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
    df['YearsSinceCreation'] = (datetime.now() - df['Created']).dt.days / 365.25
    
    return df

# Initialize Data
try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Please ensure 'roblox_games_data.csv' is in the project folder. Error: {e}")
    st.stop()

# -----------------------
# Sidebar & Context
# -----------------------
st.sidebar.title("🎮 About the Dataset")
st.sidebar.info(
    "This dataset contains 9,734 Roblox games. "
    "Analysis covers popularity, player engagement, and genre trends."
)

st.sidebar.header("Filters")
genre_filter = st.sidebar.multiselect("Select Genre", options=df["Genre"].unique(), default=df["Genre"].unique())
filtered_df = df[df["Genre"].isin(genre_filter)]

# -----------------------
# Main Dashboard Layout
# -----------------------
st.title("Roblox Games Exploratory Data Analysis")

col1, col2 = st.columns(2)

# --- Distribution Analysis ---
with col1:
    st.subheader("Distribution Analysis")
    num_col = st.selectbox("Select metric for distribution:", ['Visits', 'Likes', 'YearsSinceCreation', 'Active'])
    fig1 = px.histogram(filtered_df, x=num_col, title=f"Distribution of {num_col}", marginal="box", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig1, use_container_width=True)

# --- Scatter Analysis ---
with col2:
    st.subheader("Bivariate Analysis")
    x_axis = st.selectbox("X-Axis", ['Likes', 'YearsSinceCreation', 'Active'], key="x")
    y_axis = st.selectbox("Y-Axis", ['Visits', 'Favorites', 'Active'], key="y")
    log_scale = st.checkbox("Use Log Scale (Recommended for skew)", value=True)
    
    fig2 = px.scatter(filtered_df, x=x_axis, y=y_axis, color="Genre", hover_name="Title")
    if log_scale:
        fig2.update_xaxes(type="log")
        fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Advanced Visualizations
# -----------------------
st.divider()
st.subheader("Correlation Matrix")
numeric_df = filtered_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation between Engagement Metrics")
st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# Data Preview
# -----------------------
with st.expander("View Cleaned Raw Data"):
    st.dataframe(filtered_df)                 