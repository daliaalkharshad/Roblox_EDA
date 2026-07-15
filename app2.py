import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Roblox EDA Dashboard", layout="wide")


@st.cache_data
def load_and_clean_data():
    """Load the cleaned Roblox dataset and prepare it for analysis."""
    for path in ["Cleaned_data.csv", "roblox_games_data.csv"]:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            df = None

    if df is None:
        raise FileNotFoundError("No Roblox dataset file was found in the workspace.")

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].fillna("Unknown")
        df["Genre"] = df["Genre"].replace({"N/A": "Unknown", "": "Unknown"})

    if "Created" in df.columns:
        df["Created"] = pd.to_datetime(df["Created"], errors="coerce")
        df["YearsSinceCreation"] = (datetime.now() - df["Created"]).dt.days / 365.25

    def convert(x):
        if isinstance(x, str):
            x = x.replace("+", "").replace(",", "")
            if x.endswith("K"):
                return float(x[:-1]) * 1000
            if x.endswith("M"):
                return float(x[:-1]) * 1000000
            if x.endswith("B"):
                return float(x[:-1]) * 1000000000
            try:
                return float(x)
            except ValueError:
                return np.nan
        return x

    for col in ["Visits", "Likes", "Dislikes", "Favorites", "Active"]:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    if "Ratio_Likes" not in df.columns and "Likes" in df.columns and "Dislikes" in df.columns:
        df["Ratio_Likes"] = df["Likes"] / (df["Likes"] + df["Dislikes"]).replace(0, np.nan)
        df["Ratio_Dislikes"] = df["Dislikes"] / (df["Likes"] + df["Dislikes"]).replace(0, np.nan)

    return df


try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"The dataset could not be loaded. Please make sure one of the CSV files exists. Error: {e}")
    st.stop()


st.sidebar.title("🎮 Roblox Data Explorer")
st.sidebar.caption("A cheerful dashboard for discovering what makes Roblox games tick.")

st.sidebar.header("Filters")
genre_filter = st.sidebar.multiselect(
    "Choose genres",
    options=sorted(df["Genre"].dropna().astype(str).unique()),
    default=sorted(df["Genre"].dropna().astype(str).unique()),
)
age_filter = st.sidebar.multiselect(
    "Age recommendation",
    options=sorted(df["AgeRecommendation"].dropna().astype(str).unique()),
    default=sorted(df["AgeRecommendation"].dropna().astype(str).unique()),
)
min_visits = st.sidebar.slider("Minimum visits", min_value=0, max_value=int(df["Visits"].max()) if "Visits" in df.columns else 1000000000, value=0, step=1000000)

filtered_df = df.copy()
if genre_filter:
    filtered_df = filtered_df[filtered_df["Genre"].astype(str).isin(genre_filter)]
if age_filter:
    filtered_df = filtered_df[filtered_df["AgeRecommendation"].astype(str).isin(age_filter)]
if "Visits" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Visits"] >= min_visits]

if filtered_df.empty:
    st.warning("No games match the selected filters. Try widening the filter range.")
    st.stop()


st.markdown(
    """
    <div style="background: linear-gradient(90deg, #7C3AED 0%, #2563EB 100%); padding: 1.2rem 1.4rem; border-radius: 16px; color: white; margin-bottom: 1rem;">
        <h1 style="margin:0; font-size:2rem;">🚀 Roblox Games EDA Dashboard</h1>
        <p style="margin:0.2rem 0 0 0; font-size:1rem;">Explore the most popular titles, quirky genre patterns, and the engagement signals that stand out.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.subheader("🧾 Brief dataset summary")
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
with summary_col1:
    st.metric("Games in view", f"{len(filtered_df):,}")
with summary_col2:
    dominant_genre = filtered_df["Genre"].value_counts().idxmax() if "Genre" in filtered_df.columns else "N/A"
    st.metric("Dominant genre", dominant_genre)
with summary_col3:
    avg_visits = filtered_df["Visits"].mean() if "Visits" in filtered_df.columns else np.nan
    st.metric("Average visits", f"{avg_visits:,.0f}" if pd.notna(avg_visits) else "N/A")
with summary_col4:
    top_title = filtered_df.loc[filtered_df["Visits"].idxmax(), "Title"] if "Visits" in filtered_df.columns else "N/A"
    st.metric("Top title", top_title[:24] + ("..." if len(top_title) > 24 else ""))

st.info(
    "This dataset highlights how Roblox games differ by genre, popularity, and player sentiment. The strongest trends are around Simulation dominance, strong visit/active-player relationships, and genre-specific like/dislike behavior."
)


obs_list = [
    "Simulation is the most dominant genre in terms of total game output.",
    "Puzzle and RPG genres show the highest like ratios, while Education and Obby & Platformer show the lowest.",
    "Education and Obby & Platformer tend to have the highest dislike ratios, while Puzzle and RPG have the lowest.",
    "Active players and visits move together very strongly, showing that more engaged games attract more traffic.",
    "The median like ratio is generally high across most genres, but Simulation has a wide spread and more outliers.",
    "Most genres span a broad range of visit counts, and some categories such as Roleplay & Avatar Sim, Survival, and Action frequently produce blockbuster titles."
]

with st.expander("✨ Key EDA observations from the notebook"):
    for item in obs_list:
        st.markdown(f"- {item}")


col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribution explorer")
    metric = st.selectbox("Pick a metric", ["Visits", "Likes", "Active", "Favorites", "YearsSinceCreation"], key="dist_metric")
    fig1 = px.histogram(
        filtered_df,
        x=metric,
        title=f"Distribution of {metric}",
        marginal="box",
        color_discrete_sequence=["#7C3AED"],
        hover_data={"Title": True},
    )
    fig1.update_layout(template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("This view helps you see whether the metric is concentrated, skewed, or packed with outliers.")

with col2:
    st.subheader("🔗 Relationship explorer")
    x_axis = st.selectbox("X-axis", ["Likes", "Active", "Favorites", "YearsSinceCreation"], key="x_axis")
    y_axis = st.selectbox("Y-axis", ["Visits", "Favorites", "Active"], key="y_axis")
    log_scale = st.checkbox("Use log scale", value=True)
    scatter_args = dict(
        data_frame=filtered_df,
        x=x_axis,
        y=y_axis,
        color="Genre",
        hover_name="Title",
        template="plotly_white",
    )
    if "Visits" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["Visits"]):
        scatter_args["size"] = "Visits"
        scatter_args["size_max"] = 40

    fig2 = px.scatter(**scatter_args)
    if log_scale:
        fig2.update_xaxes(type="log")
        fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True)
    corr_value = filtered_df[[x_axis, y_axis]].corr().iloc[0, 1] if x_axis in filtered_df.columns and y_axis in filtered_df.columns else np.nan
    st.caption(f"Correlation between {x_axis} and {y_axis}: {corr_value:.2f}")


st.divider()

st.subheader("🎭 Genre spotlight")
genre_col, detail_col = st.columns([1.2, 0.8])
with genre_col:
    genre_chart = px.bar(
        filtered_df.groupby("Genre", as_index=False).size().rename(columns={"size": "Game Count"}),
        x="Genre",
        y="Game Count",
        color="Game Count",
        color_continuous_scale="Viridis",
        title="Game count by genre",
    )
    genre_chart.update_layout(template="plotly_white")
    st.plotly_chart(genre_chart, use_container_width=True)
with detail_col:
    if "Ratio_Likes" in filtered_df.columns:
        genre_like = filtered_df.groupby("Genre")["Ratio_Likes"].mean().sort_values(ascending=False).reset_index()
        st.dataframe(genre_like.head(10), use_container_width=True)
    st.caption("This chart reinforces the notebook finding that games in some genres consistently earn stronger player satisfaction than others.")


st.divider()
st.subheader("🏆 Top games by popularity")
metric_choice = st.selectbox("Highlight top games by", ["Visits", "Likes", "Favorites", "Active"], key="top_metric")
if metric_choice in filtered_df.columns:
    top_games = filtered_df.nlargest(10, metric_choice)[["Title", "Genre", metric_choice]]
    fig3 = px.bar(
        top_games,
        x="Title",
        y=metric_choice,
        color="Genre",
        title=f"Top 10 games by {metric_choice}",
        template="plotly_white",
    )
    fig3.update_xaxes(tickangle=-25)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("This section brings the biggest stars to the front so you can quickly spot blockbuster titles and their genres.")


st.divider()
st.subheader("🧠 Correlation matrix")
numeric_df = filtered_df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr = numeric_df.corr()
    fig4 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation between engagement metrics",
        template="plotly_white",
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Strong correlations suggest that certain engagement signals tend to rise together, especially visits and active users.")


with st.expander("📋 View filtered dataset"):
    st.dataframe(filtered_df[[col for col in ["Title", "Genre", "AgeRecommendation", "Visits", "Likes", "Favorites", "Active", "Created"] if col in filtered_df.columns]], use_container_width=True)
