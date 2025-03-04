import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans

# Function to load multiple CSV files
@st.cache_data
def load_data():
    file_paths = [
        "Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv",
        "Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv",
        "Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv",
        "Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv",
        "Resale flat prices based on registration date from Jan-2017 onwards.csv"
    ]

    # Load and concatenate all CSV files
    dfs = [pd.read_csv(file) for file in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert 'month' to datetime and extract year/month
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month

    # Ensure all categorical variables are string type before encoding
    categorical_columns = ['town', 'flat_type', 'street_name', 'storey_range', 'flat_model']
    for col in categorical_columns:
        df[col] = df[col].astype(str)

    return df

# Load resale price dataset
df = load_data()

# Function to load trained XGBoost model
@st.cache_data
def load_model():
    model = xgb.Booster()
    model.load_model("xgboost_market_trend_analysis.json")  
    return model

# Load the trained model
model = load_model()

# Streamlit App Layout
st.title("📊 Singapore Resale Flat Market Trend Analysis")

# Sidebar Navigation
st.sidebar.title("Navigation")
# Add Market Trend Analysis Section
option = st.sidebar.radio("Choose a section:", ["📈 Market Trends", "🏙️ Most Expensive Towns", "🛠️ Market Trend Analysis"])

# 📈 Market Trends Section (Already in Your Code)
if option == "📈 Market Trends":
    st.header("Resale Price Trends Over Time")

    yearly_trends = df.groupby('year')['resale_price'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly_trends.index, yearly_trends.values, marker='o', linestyle='-')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Resale Price')
    ax.set_title('Resale Price Trends Over Time')
    ax.grid()
    st.pyplot(fig)

# 🏙️ Most Expensive Towns Section (Already in Your Code)
elif option == "🏙️ Most Expensive Towns":
    st.header("Top 5 Most Expensive Towns Per Decade")

    df['decade'] = (df['year'] // 10) * 10

    decade_town_prices = df.groupby(['decade', 'town'])['resale_price'].mean().reset_index()
    top_towns_per_decade = decade_town_prices.sort_values(['decade', 'resale_price'], ascending=[True, False])
    top_towns_per_decade = top_towns_per_decade.groupby('decade').head(5)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_towns_per_decade, x='decade', y='resale_price', hue='town', ax=ax, palette="viridis")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Average Resale Price")
    ax.set_title("Top 5 Most Expensive Towns Per Decade (1990s - 2020s)")
    ax.legend(title="Town", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid()
    st.pyplot(fig)

# 🛠️ Market Trend Analysis (New Section)
elif option == "🛠️ Market Trend Analysis":
    st.header("📊 XGBoost Market Trend Analysis")

    # 1️⃣ Feature Importance Analysis
    st.subheader("🔍 Feature Importance (What Impacts Prices the Most?)")

    # Extract feature importance from trained model
    feature_importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({'Feature': list(feature_importance.keys()), 'Importance': list(feature_importance.values())})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature Name")
    ax.set_title("XGBoost Feature Importance for Market Trends")
    ax.invert_yaxis()
    ax.grid()
    st.pyplot(fig)

    # 2️⃣ Trend-Based Clustering (Market Segmentation)
    st.subheader("📍 Town Clustering Based on Resale Price Trends")

    # Get town-wise average resale price trends
    town_trends = df.groupby(['town'])['resale_price'].mean().reset_index()

    # Encode town names numerically
    town_trends['town_encoded'] = town_trends['town'].astype("category").cat.codes

    # Fit KMeans Clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    town_trends['Cluster'] = kmeans.fit_predict(town_trends[['town_encoded', 'resale_price']])

    # Plot Clusters
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=town_trends['town_encoded'], y=town_trends['resale_price'], hue=town_trends['Cluster'], palette='viridis', ax=ax)
    ax.set_xlabel("Town (Encoded)")
    ax.set_ylabel("Average Resale Price")
    ax.set_title("Town Clusters Based on Resale Price Trends")
    ax.grid()
    st.pyplot(fig)

    # 3️⃣ Market Anomaly Detection (Identifying Sudden Changes)
    st.subheader("⚠️ Market Anomalies (Detecting Price Spikes & Drops)")

    # Predict resale prices for historical data
    dmat = xgb.DMatrix(df.drop(columns=['resale_price']))
    df['predicted_price'] = model.predict(dmat)

    # Calculate residuals (actual - predicted)
    df['residual'] = df['resale_price'] - df['predicted_price']

    # Identify anomalies (extreme deviations)
    anomalies = df[np.abs(df['residual']) > df['residual'].std() * 2]

    # Plot Anomalies
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df['year'], df['residual'], alpha=0.5, label='Normal', color='blue')
    ax.scatter(anomalies['year'], anomalies['residual'], color='red', label='Anomalies')
    ax.axhline(0, linestyle="--", color="black")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (Price Difference)")
    ax.set_title("Market Anomalies in Resale Prices")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
