import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt

st.write("✅ Streamlit is running!")

# Load trained models
best_resale_model = joblib.load("best_resale_transaction_model.pkl")
best_price_model = joblib.load("best_price_model.pkl")
best_location_model = joblib.load("best_location_demand_model.pkl")

# Load last known lag values
last_known_lags_resale = joblib.load("last_known_lags_resale.pkl")
last_known_lags_price = joblib.load("last_known_lags_price.pkl")
last_known_lags_location = joblib.load("last_known_lags_location.pkl")

# Define available towns (match exact format used in training)
towns = [
    "Ang Mo Kio", "Bedok", "Bishan", "Bukit Batok", "Bukit Merah", "Bukit Timah",
    "Central Area", "Choa Chu Kang", "Clementi", "Geylang", "Hougang", "Jurong East",
    "Jurong West", "Kallang/Whampoa", "Marine Parade", "Pasir Ris", "Punggol",
    "Queenstown", "Sembawang", "Sengkang", "Serangoon", "Tampines", "Toa Payoh",
    "Woodlands", "Yishun"
]

# Streamlit App UI
st.title("HDB Resale Demand Forecasting 🚀")
st.markdown("This app forecasts resale transactions, prices, and demand by town.")

# Sidebar - User Input
st.sidebar.header("User Input")
year = st.sidebar.number_input("Select Year", min_value=2024, max_value=2030, value=2025)
month = st.sidebar.number_input("Select Month", min_value=1, max_value=12, value=6)
town = st.sidebar.selectbox("Select Town", towns)

# **📈 Resale Transactions Forecast**
st.subheader("📈 Resale Transactions Forecast")

# Get feature names dynamically
required_features_resale = best_resale_model.feature_names_in_

# Ensure lag values are included
X_resale = pd.DataFrame(columns=required_features_resale)
X_resale.loc[0] = [last_known_lags_resale.get(f'lag_{lag}', np.nan) for lag in [1, 3, 12]] + [year, month]

# Predict transactions
resale_forecast = best_resale_model.predict(X_resale)[0]
st.write(f"**Predicted Transactions in {year}-{month}: {int(resale_forecast)}**")

# **🏠 Resale Price Forecast**
st.subheader("🏠 Resale Price Forecast")

required_features_price = best_price_model.feature_names_in_

# Ensure lag values are included
X_price = pd.DataFrame(columns=required_features_price)
X_price.loc[0] = [last_known_lags_price.get(f'lag_{lag}', np.nan) for lag in [1, 3]] + [year, month]

# Predict price
price_forecast = best_price_model.predict(X_price)[0]
st.write(f"**Predicted Resale Price in {year}-{month}: ${price_forecast:,.2f}**")

# **🌍 Location Demand Forecast**
st.subheader("🌍 Location Demand Forecast")

# Convert town name to proper format (for matching feature names)
formatted_town = f"town_{town.replace(' ', '_')}"

# Extract feature names from the trained model
required_features_location = best_location_model.feature_names_in_

# Ensure all town columns exist
town_feature = {col: 0 for col in required_features_location}  # Create zero-filled dictionary
if formatted_town in town_feature:
    town_feature[formatted_town] = 1  # Set selected town to 1

# Convert dictionary to DataFrame with correct columns
X_location = pd.DataFrame([{**town_feature}])

# Predict location demand
location_forecast = best_location_model.predict(X_location)[0]
st.write(f"**Predicted Demand in {town} for {year}-{month}: {int(location_forecast)}**")

# **📊 Forecasted Trends Visualization**
fig, ax = plt.subplots(figsize=(10, 4))
dates = pd.date_range(start="2024-01-01", periods=24, freq='M')

# Fix issue in prediction loop (ensure lag features are included)
X_resale_series = pd.DataFrame(columns=required_features_resale)
for d in dates:
    X_resale_series.loc[len(X_resale_series)] = [last_known_lags_resale.get(f'lag_{lag}', np.nan) for lag in [1, 3, 12]] + [d.year, d.month]

ax.plot(dates, best_resale_model.predict(X_resale_series), label="Resale Transactions")
ax.plot(dates, [best_price_model.predict(pd.DataFrame([[d.year, d.month]], columns=['year', 'month_num']))[0] for d in dates], label="Resale Prices")
ax.plot(dates, [best_location_model.predict(pd.DataFrame([[d.year, d.month] + list(town_feature.values())], columns=['year', 'month_num'] + list(town_feature.keys())))[0] for d in dates], label=f"Demand in {town}")

ax.set_xlabel("Date")
ax.set_ylabel("Forecasted Values")
ax.legend()
st.pyplot(fig)

st.write("💡 **Note**: The forecasts are based on the trained models and historical patterns.")
