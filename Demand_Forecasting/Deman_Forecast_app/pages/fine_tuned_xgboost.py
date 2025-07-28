import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import boto3
from io import StringIO
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
st.image("model_metrics.png", caption="Evaluation Metrics and SHAP Summary")
st.subheader("📅 Monthly Aggregated – Actual vs Predicted")
st.image("Total_VS_Predicted.png")
st.subheader("📍 Actual vs Predicted Scatter Plot")
st.image("Actual_VS_Predicted_Scatterplot.png")

st.subheader("🌇️ Town-Level Prediction")

s3 = boto3.client('s3')
# Helper function to load a CSV from S3
def load_csv_from_s3(bucket_name, key):
    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(content))
bucket = 'aop-demand-forecast-dataset'

# === Load Model and Stats ===
model_bundle = joblib.load("xgboost_finetune_model.pkl")
model = model_bundle["model"]
features = model_bundle["features"]

# === Load and Prepare Historical Data ===
data = load_csv_from_s3(bucket,"merged_hdb_resale_data_up_to_may2025_cleaned.csv")
data['month'] = pd.to_datetime(data['month'])
data = data.groupby(['month', 'town']).size().reset_index(name='transaction_count')
data = data.sort_values(['town', 'month']).copy()

def create_features(df):
    df = df.copy()
    df["month_num"] = df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    g = df.groupby("town")["transaction_count"]
    df["lag_1"] = g.shift(1)
    df["lag_3"] = g.shift(3)
    df["lag_12"] = g.shift(12)
    df["lag_ratio"] = df["lag_1"] / (df["lag_3"] + 1e-6)
    df["yoy_lag_12"] = (df["transaction_count"] - df["lag_12"]) / (df["lag_12"] + 1e-6)
    df["local_trend_3m"] = df["lag_1"] - df["lag_3"]

    return df

history = create_features(data.copy())
history.dropna(inplace=True)

# === Cache Forecast ===
if "forecast_df" not in st.session_state:
    future_predictions = []
    future_months = pd.date_range("2025-06-01", "2025-12-01", freq="MS")
    towns = history["town"].unique()

    for town in towns:
        town_df = history[history["town"] == town].copy()
        town_df = town_df[["month", "transaction_count", "town"]].sort_values("month")

        for future_month in future_months:
            lag_data = {}
            for lag in [1, 3, 12]:
                lag_month = future_month - relativedelta(months=lag)
                val = town_df[town_df["month"] == lag_month]["transaction_count"]
                lag_data[f"lag_{lag}"] = val.values[0] if not val.empty else np.nan

            if any(pd.isna(v) for v in lag_data.values()):
                continue

            month_num = future_month.month
            row = {
                "month_num": month_num,
                "month_sin": np.sin(2 * np.pi * month_num / 12),
                "month_cos": np.cos(2 * np.pi * month_num / 12),
                **lag_data,
                "lag_ratio": lag_data["lag_1"] / (lag_data["lag_3"] + 1e-6),
                "yoy_lag_12": (lag_data["lag_1"] - lag_data["lag_12"]) / (lag_data["lag_12"] + 1e-6),
                "local_trend_3m": lag_data["lag_1"] - lag_data["lag_3"],
                f"town_{town}": 1
            }

            row_df = pd.DataFrame([row], columns=features).fillna(0)
            y_pred = model.predict(row_df)[0]

            future_predictions.append({
                "month": future_month,
                "town": town,
                "predicted_transaction_count": round(y_pred, 2)
            })

            town_df = pd.concat([town_df, pd.DataFrame([{
                "month": future_month,
                "transaction_count": y_pred,
                "town": town
            }])], ignore_index=True)

    future_df = pd.DataFrame(future_predictions)
    future_df["year"] = future_df["month"].dt.year
    future_df["month_num"] = future_df["month"].dt.month
    future_df["month_name"] = future_df["month"].dt.strftime("%B")

    hist_df = load_csv_from_s3(bucket,"all_historical_predictions.csv")
    hist_df["month"] = pd.to_datetime(hist_df["month"])
    hist_df["year"] = hist_df["month"].dt.year
    hist_df["month_num"] = hist_df["month"].dt.month
    hist_df["month_name"] = hist_df["month"].dt.strftime("%B")
    hist_df = hist_df.rename(columns={"town_raw": "town"})

    forecast_df = pd.concat([hist_df, future_df], ignore_index=True)
    forecast_df["predicted_transaction_count"] = forecast_df["predicted_transaction_count"].astype(int)
    st.session_state["forecast_df"] = forecast_df

# === UI Filters ===
st.markdown("### 🔍 Filter Forecasts")

with st.form("forecast_form"):
    forecast_df = st.session_state["forecast_df"]
    selected_town = st.selectbox("🏡 Select a Town", sorted(forecast_df["town"].unique()))
    
    year_options = sorted(forecast_df["year"].unique())
    start_year = st.selectbox("📅 Start Year", year_options, index=0)
    end_year = st.selectbox("📅 End Year", year_options, index=len(year_options)-1)

    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]

    start_month_name = st.selectbox("🟢 Start Month", month_names, index=0)
    end_month_name = st.selectbox("🔴 End Month", month_names, index=11)

    submitted = st.form_submit_button("🚀 Predict")

if submitted:
    start_month_num = month_names.index(start_month_name) + 1
    end_month_num = month_names.index(end_month_name) + 1
    start_month = pd.to_datetime(f"{start_year}-{start_month_num:02d}-01")
    end_month = pd.to_datetime(f"{end_year}-{end_month_num:02d}-01")



    if start_month > end_month:
        st.warning("⚠️ Start Month must be before End Month.")
    else:
        filtered_df = forecast_df[
            (forecast_df["town"] == selected_town) &
            (forecast_df["month"] >= start_month) &
            (forecast_df["month"] <= end_month)
        ].copy()

        date_range = pd.date_range(start_month, end_month, freq="MS")
        full_month_df = pd.DataFrame({"month": date_range})
        full_month_df["month_num"] = full_month_df["month"].dt.month
        full_month_df["month_name"] = full_month_df["month"].dt.strftime("%B")

        full_month_df = full_month_df.merge(
            filtered_df[["month", "predicted_transaction_count"]],
            on="month", how="left"
        )
        full_month_df["predicted_transaction_count"] = full_month_df["predicted_transaction_count"].fillna(0).astype(int)

        st.markdown(f"### 📈 Predicted Transactions in {selected_town} from {start_month.strftime('%B')} to {end_month.strftime('%B')}")
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(full_month_df["month"], full_month_df["predicted_transaction_count"], marker='o')
        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted Transaction Count")
        full_month_df["month_str"] = full_month_df["month"].dt.strftime("%Y-%m")
        ax.set_xticks(full_month_df["month"])
        ax.set_xticklabels(full_month_df["month_str"], rotation=45)
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("### 📋 Forecast Table")
        table_df = full_month_df.copy()
        table_df["Year"] = table_df["month"].dt.year
        table_df["Month"] = table_df["month"].dt.strftime("%Y-%m")
        table_df["Town"] = selected_town

        table_df.rename(columns={
            "month_num": "Month (Num)",
            "month_name": "Month (Name)",
            "predicted_transaction_count": "Predicted Transaction Count"
        }, inplace=True)

        st.dataframe(table_df[["Year", "Month", "Month (Num)", "Month (Name)", "Town", "Predicted Transaction Count"]])
