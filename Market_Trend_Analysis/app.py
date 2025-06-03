import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import mlflow

mlflow.set_experiment("AOP_Market_Trend_Analysis")
st.set_page_config(page_title="AOP Market Trend Analysis Tool", layout="wide")

# --- Label Encoders ---
town_classes = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON',
                'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
le_town = LabelEncoder().fit(town_classes)

flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
le_flat_type = LabelEncoder().fit(flat_types)

storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21',
                 '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42',
                 '43 TO 45', '46 TO 48', '49 TO 51']
le_storey_range = LabelEncoder().fit(storey_ranges)

flat_models = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED', 'MAISONETTE',
               'APARTMENT', 'MODEL A-MAISONETTE', 'TERRACE', '2-ROOM', 'DBSS', 'IMPROVED-MAISONETTE',
               'MODEL A2', 'TYPE S1', 'TYPE S2', 'PREMIUM APARTMENT', 'MULTI GENERATION']
le_flat_model = LabelEncoder().fit(flat_models)

# --- Load Property Info and Resale Index Data ---
@st.cache_data
def load_property_info():
    df_property = pd.read_csv('HDBPropertyInformation.csv')
    df_property['block'] = df_property['blk_no'].astype(str).str.strip().str.upper()
    df_property['street_name'] = df_property['street'].astype(str).str.strip().str.upper()
    return df_property

@st.cache_data
def load_resale_index():
    df_resale_idx = pd.read_csv('HDBResalePriceIndex1Q2009100Quarterly.csv')
    return df_resale_idx

df_property = load_property_info()
df_resale_index = load_resale_index()

# --- Load Model ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_market_trend_analysis.json")
    return model

model = load_model()

# --- Load Data ---
@st.cache_data
def load_full_data():
    dfs = [
        pd.read_csv(f) for f in [
            'Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv',
            'Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv',
            'Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv',
            'Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv',
            'Resale flat prices based on registration date from Jan-2017 onwards.csv']
    ]
    df = pd.concat(dfs, ignore_index=True)
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['town'] = df['town'].str.upper()
    return df

df_full = load_full_data()

# --- Sidebar Prediction Tool ---
st.sidebar.header("🔍 Predict Resale Price")
town = st.sidebar.selectbox("Town", town_classes)
flat_type = st.sidebar.selectbox("Flat Type", flat_types)
storey_range = st.sidebar.selectbox("Storey Range", storey_ranges)
flat_model = st.sidebar.selectbox("Flat Model", flat_models)
floor_area = st.sidebar.number_input("Floor Area (sqm)", min_value=30, max_value=200, value=80)
lease_commence = st.sidebar.number_input("Lease Commence Year", min_value=1960, max_value=2023, value=2000)
remaining_lease = st.sidebar.number_input("Remaining Lease (Years)", min_value=1, max_value=99, value=78)
year = st.sidebar.selectbox("Transaction Year", list(range(1990, 2024)), index=33)
month_num = st.sidebar.selectbox("Transaction Month", list(range(1, 13)), index=0)

quarter = ((month_num - 1) // 3) + 1

# Calculate quarter label
quarter_label = f"{year}Q{quarter}"

property_match = df_property

# Since property dae doesn't have town / flat_type info, just use default:
if not df_property.empty:
    year_completed = int(df_property['year_completed'].median())
    total_dwelling_units = int(df_property['total_dwelling_units'].median())
else:
    year_completed = lease_commence
    total_dwelling_units = 0

building_age = year - year_completed

# Lookup resale index:
index_match = df_resale_index[df_resale_index['quarter'] == quarter_label]
if not index_match.empty:
    resale_index_value = float(index_match['index'].iloc[0])
else:
    resale_index_value = df_resale_index['index'].mean()

# Build model input:
encoded_input = pd.DataFrame([{
    'town': le_town.transform([town])[0],
    'flat_type': le_flat_type.transform([flat_type])[0],
    'street_name': 0,
    'storey_range': le_storey_range.transform([storey_range])[0],
    'floor_area_sqm': floor_area,
    'flat_model': le_flat_model.transform([flat_model])[0],
    'lease_commence_date': lease_commence,
    'remaining_lease': remaining_lease,
    'year_completed': year_completed,
    'total_dwelling_units': total_dwelling_units,
    'year': year,
    'month_num': month_num,
    'quarter': quarter,
    'building_age': building_age,
    'index': resale_index_value
}])

if st.sidebar.button("💡 Predict Price"):
    predicted_price = model.predict(encoded_input)[0]
    st.sidebar.success(f"💰 SGD ${predicted_price:,.2f}")
    with mlflow.start_run():
        mlflow.log_params(encoded_input.to_dict('records')[0])
        mlflow.log_metric("predicted_price", predicted_price)

# --- Dashboard View ---
st.title("📊 AOP Resale Market Dashboard")
col1, col2 = st.columns(2)

# --- Yearly Average Price Trend ---
with col1:
    st.subheader("📈 Avg Resale Price Over Years")
    st.markdown("_This graph shows the average resale prices of flats across all towns in Singapore over the years._")
    yearly_avg = df_full.groupby('year')['resale_price'].mean().reset_index()
    st.line_chart(yearly_avg.set_index('year'))

# --- Town Comparison ---
with col2:
    st.subheader("🏙️ Top 10 Most Expensive Towns")
    st.markdown("_This bar chart highlights the 10 towns with the highest average resale flat prices, helping users identify premium residential areas._")
    town_avg = df_full.groupby('town')['resale_price'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(town_avg)

# --- Flat Type Trend ---
st.subheader("🧱 Flat Type Trends")
st.markdown("_This chart compares the average resale price trends of different flat types (e.g., 3-room, 4-room) over the years._")
flat_type_trend = df_full.groupby(['year', 'flat_type'])['resale_price'].mean().reset_index()
flat_type_trend_pivot = flat_type_trend.pivot(index='year', columns='flat_type', values='resale_price')
st.line_chart(flat_type_trend_pivot)

# --- Volatility ---
st.subheader("🔄 Price Volatility Over Time")
st.markdown("_Shows the month-to-month variation in resale prices, indicating how stable or volatile the market has been._")
df_full['year_month'] = df_full['month'].dt.to_period('M').astype(str)
monthly_std = df_full.groupby('year_month')['resale_price'].std().reset_index()
st.line_chart(monthly_std.set_index('year_month'))

# --- Heatmap (Text View Only) ---
st.subheader("🗺️ Regional Heatmap Table")
st.markdown("_This heatmap provides a year-by-year view of average resale prices for each town, helping to spot patterns across time and geography._")
heatmap_data = df_full.groupby(['year', 'town'])['resale_price'].mean().unstack().fillna(0)
st.dataframe(heatmap_data.style.background_gradient(cmap='YlGnBu', axis=1))

# --- CAGR ---
st.subheader("⏳ Town-Level CAGR")
st.markdown("_Calculates the Compound Annual Growth Rate (CAGR) in average resale price for each town from 2013 to the most recent year._")
base_year = 2013
latest_year = df_full['year'].max()
town_avg_base = df_full[df_full['year'] == base_year].groupby('town')['resale_price'].mean()
town_avg_latest = df_full[df_full['year'] == latest_year].groupby('town')['resale_price'].mean()
appreciation_df = pd.DataFrame({
    'Base Year Price (2013)': town_avg_base,
    f'Latest Year Price ({latest_year})': town_avg_latest
}).dropna()
appreciation_df['CAGR (%)'] = ((appreciation_df[f'Latest Year Price ({latest_year})'] /
                                appreciation_df['Base Year Price (2013)']) ** (1 / (latest_year - base_year)) - 1) * 100
appreciation_df = appreciation_df.sort_values(by='CAGR (%)', ascending=False)
st.dataframe(appreciation_df.style.format({
    'Base Year Price (2013)': 'S${:,.0f}',
    f'Latest Year Price ({latest_year})': 'S${:,.0f}',
    'CAGR (%)': '{:.2f}%'
}))

# --- Volume ---
st.subheader("📊 Annual Transactions Volume")
st.markdown("_Displays the total number of resale transactions each year, providing insight into market activity and demand trends._")
volume_by_year = df_full.groupby('year').size().reset_index(name='Transactions')
st.bar_chart(volume_by_year.set_index('year'))

# --- Model Comparison ---
st.subheader("🤖 Model Comparison Across Algorithms")
st.markdown("_This section compares three different predictive models (XGBoost, LSTM, ARIMA) based on error metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). It helps evaluate which model is more accurate and consistent for forecasting resale flat prices._")

model_comparison = pd.DataFrame({
    "Model": ["XGBoost", "LSTM", "ARIMA"],
    "MAE": [13241.14, 16454.58, 69533.19],
    "RMSE": [19265.25, 18599.16, 19361.81],
    "MAPE": ["4.76%", "3.07%", "12.59%"]
})

st.subheader("📋 Model Evaluation Metrics")
st.table(model_comparison)

st.subheader("📷 Actual vs Predicted Comparison")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("XGBoost_AVP.png", caption="XGBoost Actual vs Predicted", use_container_width=True)

with col2:
    st.image("LSTM_AVP.png", caption="LSTM Actual vs Predicted", use_container_width=True)

with col3:
    st.image("ARIMA_AVP.png", caption="ARIMA Actual vs Predicted", use_container_width=True)
