import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# === Streamlit App Config ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 95%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Fine-Tuned XGBoost Model Evaluation")
st.image("images/model_metrics.png", caption="Evaluation Metrics and SHAP Summary")
st.subheader("📅 Monthly Aggregated – Actual vs Predicted")
st.image("images/Total_VS_Predicted.png")
st.subheader("📍 Actual vs Predicted Scatter Plot")
st.image("images/Actual_VS_Predicted_Scatterplot.png")


st.subheader("🌇️ Town-Level Prediction")

# === Load Model and Stats ===
model_bundle = joblib.load("models/demandforecasting_best_xgb_model_finetuned_V10.pkl")
model = model_bundle["model"]
features = model_bundle["features"]
norm_stats = model_bundle["normalization_stats"]

# === Load and Prepare Historical Data ===
data = pd.read_csv("data/merged_hdb_resale_data_up_to_may2025_cleaned.csv")
data['month'] = pd.to_datetime(data['month'])
data = data.groupby(['month', 'town']).size().reset_index(name='transaction_count')
data = data.sort_values(['town', 'month']).copy()

def create_features(df):
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    g = df.groupby('town')['transaction_count']
    df['rolling_avg'] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    df['rolling_std'] = g.transform(lambda x: x.shift(1).rolling(3).std())
    df['lag_1'] = g.shift(1)
    df['lag_3'] = g.shift(3)
    df['lag_6'] = g.shift(6)
    df['lag_12'] = g.shift(12)
    df['lag_ratio'] = df['lag_1'] / df['lag_3']
    df['yoy_lag_12'] = (df['transaction_count'] - df['lag_12']) / df['lag_12']
    df['local_trend_3m'] = df['lag_1'] - df['lag_3']
    df['residual_trend_3m'] = g.transform(lambda x: x.shift(1) - x.shift(1).rolling(3).mean())

    tv_map = df.groupby("town")["transaction_count"].mean().to_dict()
    df["town_volume"] = df["town"].map(tv_map)
    df["town_avg_ratio"] = df["transaction_count"] / df["town_volume"]
    df["town_rank"] = df["town_volume"].rank()
    df["town_longterm_dev"] = df["transaction_count"] - df["town_volume"]
    df["volatility"] = df["rolling_std"] / (df["rolling_avg"] + 1e-6)
    df["is_high_volume"] = (df["town_volume"] > df["town_volume"].median()).astype(int)
    df["is_volatile"] = (df["rolling_std"] > df["rolling_std"].median()).astype(int)

    return df

history = create_features(data.copy())
history.dropna(inplace=True)

# === Predict for all months in dataset and future ===
predictions = []
towns = history["town"].unique()
all_months = history["month"].unique()

# === Load Full Historical Predictions ===
forecast_df = pd.read_csv("data/all_historical_predictions.csv")
forecast_df["month"] = pd.to_datetime(forecast_df["month"])
forecast_df["year"] = forecast_df["month"].dt.year
forecast_df["month_num"] = forecast_df["month"].dt.month
forecast_df["month_name"] = forecast_df["month"].dt.strftime("%B")
forecast_df = forecast_df.rename(columns={"town_raw": "town", "predicted_transaction_count": "predicted_transaction_count"})

# === Predict for Future Months ONLY ===
future_predictions = []
future_months = pd.date_range("2025-06-01", "2025-12-01", freq="MS")
towns = forecast_df["town"].unique()

for town in towns:
    town_df = history[history["town"] == town].copy()
    for month in future_months:
        temp_df = town_df.copy()
        temp_df = pd.concat([temp_df, pd.DataFrame({"month": [month], "town": [town]})], ignore_index=True)
        temp_df = create_features(temp_df)
        row = temp_df[temp_df["month"] == month].copy().fillna(0)
        row = pd.get_dummies(row, columns=["town"])
        for col in features:
            if col not in row.columns:
                row[col] = 0
        row = row[features]
        for col in norm_stats:
            if col in row.columns:
                row[col] = (row[col] - norm_stats[col]['mean']) / norm_stats[col]['std']
        y_pred = model.predict(row)[0]
        future_predictions.append({
            "month": month,
            "town": town,
            "predicted_transaction_count": round(y_pred, 2)
        })

# === Merge Historical + Future ===
future_df = pd.DataFrame(future_predictions)
future_df["year"] = future_df["month"].dt.year
future_df["month_num"] = future_df["month"].dt.month
future_df["month_name"] = future_df["month"].dt.strftime("%B")

forecast_df = pd.concat([forecast_df, future_df], ignore_index=True)
forecast_df["predicted_transaction_count"] = forecast_df["predicted_transaction_count"].astype(int)


# === UI Filters ===
# === Updated Month Filters ===
st.markdown("### 🔍 Filter Forecasts")

selected_town = st.selectbox("🏠 Select a Town", sorted(forecast_df["town"].unique()))
selected_year = st.selectbox("📅 Select Year", sorted(forecast_df["year"].unique()))

month_names = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]

start_month_name = st.selectbox("🟢 Start Month", month_names, index=0)
end_month_name = st.selectbox("🔴 End Month", month_names, index=11)


# Convert to numeric month for filtering
start_month_num = month_names.index(start_month_name) + 1
end_month_num = month_names.index(end_month_name) + 1


# === Predict button ===
if st.button("🚀 Predict"):
    start_month = pd.to_datetime(f"{selected_year}-{start_month_num:02d}-01")
    end_month = pd.to_datetime(f"{selected_year}-{end_month_num:02d}-01")


    if start_month > end_month:
        st.warning("⚠️ Start Month must be before End Month.")
    else:
        # === Filter future forecast range ===
        filtered_df = forecast_df[
            (forecast_df["town"] == selected_town) &
            (forecast_df["month"] >= start_month) &
            (forecast_df["month"] <= end_month)
        ].copy()

        # === Ensure all months present ===
        date_range = pd.date_range(start_month, end_month, freq="MS")
        full_month_df = pd.DataFrame({"month": date_range})
        full_month_df["month_num"] = full_month_df["month"].dt.month
        full_month_df["month_name"] = full_month_df["month"].dt.strftime("%B")

        full_month_df = full_month_df.merge(
            filtered_df[["month", "predicted_transaction_count"]],
            on="month", how="left"
        )
        full_month_df["predicted_transaction_count"] = full_month_df["predicted_transaction_count"].fillna(0).astype(int)

        # === Plot ===
        st.markdown(f"### 📈 Predicted Transactions in {selected_town} from {start_month.strftime('%B')} to {end_month.strftime('%B')}")

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(full_month_df["month"], full_month_df["predicted_transaction_count"], marker='o')
        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted Transaction Count")
        ax.set_xticks(full_month_df["month"])
        ax.set_xticklabels(full_month_df["month_name"], rotation=45)
        ax.grid(True)
        st.pyplot(fig)

        # === Table ===
        st.markdown("### 📋 Forecast Table")
        table_df = full_month_df.copy()
        table_df["Year"] = table_df["month"].dt.year
        table_df["Town"] = selected_town
        table_df.rename(columns={
            "month": "Month",
            "month_num": "Month (Num)",
            "month_name": "Month (Name)",
            "predicted_transaction_count": "Predicted Transaction Count"
        }, inplace=True)
        st.dataframe(table_df[["Year", "Month", "Month (Num)", "Month (Name)", "Town", "Predicted Transaction Count"]])
