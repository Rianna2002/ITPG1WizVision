import streamlit as st
import pandas as pd
import numpy as np
import joblib
from dateutil.relativedelta import relativedelta
import boto3
from io import StringIO
st.title("📈 XGBoost Monthly Transaction Predictions per Town")

st.markdown("""
    <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 95%;
        }
    </style>
""", unsafe_allow_html=True)
s3 = boto3.client('s3')
# Helper function to load a CSV from S3
def load_csv_from_s3(bucket_name, key):
    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(content))
bucket = 'aop-demand-forecast-dataset'
@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.pkl")

@st.cache_data
def load_raw_data():
    df = load_csv_from_s3(bucket,"merged_hdb_resale_data.csv")
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
    return df

model = load_model()
raw_data = load_raw_data()

# === Historical Data Preparation ===
data = raw_data.groupby(['month', 'town']).size().reset_index(name='transaction_count')
lags = [1, 3, 6]
for lag in lags:
    data[f'lag_{lag}'] = data.groupby('town')['transaction_count'].shift(lag)
data = data.dropna()
data = pd.get_dummies(data, columns=['town'])
X = data.drop(['transaction_count', 'month'], axis=1)
data['xgb_pred'] = model.predict(X)

display_df = data[['month']].copy()
display_df['town'] = raw_data.sort_values(['month', 'town']).drop_duplicates(['month', 'town'], keep='last').reset_index(drop=True)['town']
display_df['xgb_pred'] = data['xgb_pred'].values
display_df['year'] = display_df['month'].dt.year.astype(int)
display_df['month_name'] = display_df['month'].dt.strftime('%B')
display_df['month_num'] = display_df['month'].dt.month

# === Future Forecasting (Auto-regressive) ===
history = display_df[['month', 'town', 'xgb_pred']].rename(columns={'xgb_pred': 'transaction_count'})
history = history.sort_values(['town', 'month'])

future_months = pd.date_range(start="2025-03-01", end="2025-12-01", freq='MS')
future_predictions = []

for town in history['town'].unique():
    town_df = history[history['town'] == town].copy()

    for future_month in future_months:
        lag_vals = {}
        for lag in lags:
            lag_month = future_month - relativedelta(months=lag)
            val = town_df[town_df['month'] == lag_month]['transaction_count']
            lag_vals[f'lag_{lag}'] = val.values[0] if not val.empty else np.nan

        if any(pd.isna(v) for v in lag_vals.values()):
            continue

        feature_row = {**lag_vals, f'town_{town}': 1}
        X_pred = pd.DataFrame([feature_row], columns=model.feature_names_in_).fillna(0)
        pred = model.predict(X_pred)[0]

        future_predictions.append({
            'month': future_month,
            'town': town,
            'Predicted Transaction Count': round(pred),
            'year': future_month.year,
            'month_num': future_month.month,
            'month_name': future_month.strftime('%B')
        })

        # Append to town_df to allow next month prediction
        town_df = pd.concat([town_df, pd.DataFrame([{
            'month': future_month,
            'transaction_count': pred
        }])])

future_df = pd.DataFrame(future_predictions)

# === Combine with Historical Predictions ===
display_df['Predicted Transaction Count'] = display_df['xgb_pred'].round().astype(int)
display_df = display_df[['month', 'town', 'year', 'month_num', 'month_name', 'Predicted Transaction Count']]

combined = pd.concat([display_df, future_df], ignore_index=True)

# === Sidebar ===
selected_town = st.selectbox("🏩 Select a Town", sorted(combined['town'].unique()), index=sorted(combined['town'].unique()).index("BUKIT PANJANG") if "BUKIT PANJANG" in combined['town'].unique() else 0)
selected_year = st.selectbox("🗓 Select Year", sorted(combined['year'].unique(), reverse=True), index=list(sorted(combined['year'].unique(), reverse=True)).index(2024) if 2024 in combined['year'].unique() else 0)

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
avail_months = combined[combined['year'] == selected_year]['month_name'].unique()
avail_months = [m for m in month_order if m in avail_months]
selected_month = st.selectbox("🗓 Select Month (or All)", ['All'] + avail_months)

# === Filter and Display ===
filtered = combined[
    (combined['town'] == selected_town) &
    (combined['year'] == selected_year)
]
if selected_month != "All":
    filtered = filtered[filtered['month_name'] == selected_month]

filtered = filtered.sort_values(by='month_num')

st.subheader(f"Predicted Transactions in {selected_town} for {selected_year}{' - ' + selected_month if selected_month != 'All' else ''}")
st.line_chart(
    data=filtered.set_index(filtered['month_num'])['Predicted Transaction Count']
)

table_df = filtered[['year', 'month_num', 'month_name', 'town', 'Predicted Transaction Count']].rename(
    columns={
        'year': 'Year',
        'month_num': 'Month (Num)',
        'month_name': 'Month (Name)',
        'town': 'Town'
    }
)

# 💡 Fix comma issue by converting to string
table_df['Year'] = table_df['Year'].astype(str)

st.dataframe(table_df.reset_index(drop=True))
