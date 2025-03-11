import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import pickle

# Streamlit App Title
st.title("📊 Resale Flat Price Prediction & Analysis")

# Sidebar Navigation
st.sidebar.title("Market Trend Analysis")
option = st.sidebar.radio("Choose a section:", ["📊 Model Comparison", "📈 Market Trends", "🏙️ Most Expensive Towns"])

# Load Data
@st.cache_data
def load_data():
    file1 = pd.read_csv('Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv')
    file2 = pd.read_csv('Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv')
    file3 = pd.read_csv('Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv')
    file4 = pd.read_csv('Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv')
    file5 = pd.read_csv('Resale flat prices based on registration date from Jan-2017 onwards.csv')

    for df in [file1, file2, file3]:
        df['remaining_lease'] = np.nan

    df = pd.concat([file1, file2, file3, file4, file5], ignore_index=True)

    # Feature Engineering
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df.drop(columns=['month'], inplace=True)

    # Label Encoding
    le_town = LabelEncoder()
    le_street = LabelEncoder()
    le_flat_type = LabelEncoder()
    le_storey_range = LabelEncoder()
    le_flat_model = LabelEncoder()

    df['town'] = le_town.fit_transform(df['town'])
    df['street_name'] = le_street.fit_transform(df['street_name'])
    df['flat_type'] = le_flat_type.fit_transform(df['flat_type'])
    df['storey_range'] = le_storey_range.fit_transform(df['storey_range'])
    df['flat_model'] = le_flat_model.fit_transform(df['flat_model'])

    return df, le_town

df, le_town = load_data()

# Define Target & Features
y = df['resale_price']
X = df.drop(columns=['resale_price', 'block'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert 'remaining_lease' to numeric
X_train['remaining_lease'] = X_train['remaining_lease'].astype(str).str.extract('(\d+)').astype(float)
X_test['remaining_lease'] = X_test['remaining_lease'].astype(str).str.extract('(\d+)').astype(float)

# Train Models (Only Runs Once - Cache for Performance)
@st.cache_resource
def train_models():
    # XGBoost Model
    XGB_model = XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=11, subsample=0.9,
        colsample_bytree=0.9, gamma=0.1, reg_lambda=0.7, reg_alpha=0.3, random_state=42
    )
    XGB_model.fit(X_train, y_train)
    XGB_y_pred = XGB_model.predict(X_test)
    XGB_mae = mean_absolute_error(y_test, XGB_y_pred)

    # LightGBM Model
    LGBM_model = LGBMRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=11, subsample=0.9,
        colsample_bytree=0.9, min_gain_to_split=0.1, reg_lambda=0.7, reg_alpha=0.3, random_state=42
    )
    LGBM_model.fit(X_train, y_train)
    LGBM_y_pred = LGBM_model.predict(X_test)
    LGBM_mae = mean_absolute_error(y_test, LGBM_y_pred)

    # Random Forest Model
    RF_model = RandomForestRegressor(
        n_estimators=1200, max_depth=11, max_features=0.9,
        min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42
    )
    RF_model.fit(X_train, y_train)
    RF_y_pred = RF_model.predict(X_test)
    RF_mae = mean_absolute_error(y_test, RF_y_pred)

    return XGB_model, XGB_mae, XGB_y_pred, LGBM_model, LGBM_mae, LGBM_y_pred, RF_model, RF_mae, RF_y_pred

# Load trained models and predictions
XGB_model, XGB_mae, XGB_y_pred, LGBM_model, LGBM_mae, LGBM_y_pred, RF_model, RF_mae, RF_y_pred = train_models()

# 📊 Model Comparison Page
if option == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")

    # Display MAE values in a table
    mae_values = {
        "XGBoost": XGB_mae,
        "Random Forest": RF_mae,
        "LightGBM": LGBM_mae
    }

    mae_df = pd.DataFrame(mae_values.items(), columns=["Model", "Mean Absolute Error (MAE)"])
    
    # Show table
    st.table(mae_df)

    # Show bar chart comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=mae_df["Model"], y=mae_df["Mean Absolute Error (MAE)"], palette='coolwarm', ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Absolute Error (Lower is Better)")
    ax.set_title("Model MAE Comparison")
    ax.bar_label(ax.containers[0])  # Show values on bars
    st.pyplot(fig)

    # 📌 Show Actual vs Predicted Scatter Plots (Side by Side)
    st.subheader("📈 Actual vs Predicted Resale Prices")

    col1, col2, col3 = st.columns(3)  # Create three columns

    with col1:
        st.subheader("XGBoost")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, XGB_y_pred, alpha=0.5, color="blue")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
        ax.set_xlabel("Actual Resale Price")
        ax.set_ylabel("XGB Predicted Resale Price")
        ax.set_title("XGB Actual vs Predicted")
        ax.grid()
        st.pyplot(fig)

    with col2:
        st.subheader("LightGBM")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, LGBM_y_pred, alpha=0.5, color="blue")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
        ax.set_xlabel("Actual Resale Price")
        ax.set_ylabel("LGBM Predicted Resale Price")
        ax.set_title("LGBM Actual vs Predicted")
        ax.grid()
        st.pyplot(fig)

    with col3:
        st.subheader("Random Forest")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, RF_y_pred, alpha=0.5, color="blue")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
        ax.set_xlabel("Actual Resale Price")
        ax.set_ylabel("RF Predicted Resale Price")
        ax.set_title("RF Actual vs Predicted")
        ax.grid()
        st.pyplot(fig)

# Market Trends Section
elif option == "📈 Market Trends":
    st.header("Resale Price Trends Over Time")

    yearly_trends = df.groupby('year')['resale_price'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly_trends.index, yearly_trends.values, marker='o', linestyle='-')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Resale Price')
    ax.set_title('Resale Price Trends Over Time')
    ax.grid()
    st.pyplot(fig)

# 🏙️ Most Expensive Towns Section
elif option == "🏙️ Most Expensive Towns":
    st.header("🏙️ Most Expensive Towns Analysis")

    # 📌 Part 1: Top 10 Most Expensive Towns Overall
    st.subheader("🏆 Top 10 Most Expensive Towns Overall")

    # Group data by town and calculate average resale price
    town_trends = df.groupby('town')['resale_price'].mean().sort_values(ascending=False).head(10)

    # Convert town numbers back to names
    town_trends.index = town_trends.index.map(lambda x: le_town.inverse_transform([x])[0])

    # Plot top 10 towns
    fig, ax = plt.subplots(figsize=(12, 6))
    town_trends.plot(kind='bar', color='blue', ax=ax)
    ax.set_xlabel("Town")
    ax.set_ylabel("Average Resale Price")
    ax.set_title("Top 10 Most Expensive Towns for Resale Flats")
    ax.set_xticklabels(town_trends.index, rotation=45)
    ax.grid()
    st.pyplot(fig)

    # 📌 Part 2: Top 5 Most Expensive Towns Per Decade
    st.subheader("📅 Top 5 Most Expensive Towns Per Decade")

    df['decade'] = (df['year'] // 10) * 10
    decade_town_prices = df.groupby(['decade', 'town'])['resale_price'].mean().reset_index()
    top_towns_per_decade = decade_town_prices.sort_values(['decade', 'resale_price'], ascending=[True, False])
    top_towns_per_decade = top_towns_per_decade.groupby('decade').head(5)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_towns_per_decade, x='decade', y='resale_price', hue='town', ax=ax, palette="viridis")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Average Resale Price")
    ax.set_title("Top 5 Most Expensive Towns Per Decade (1990s - 2020s)")
    st.pyplot(fig)

