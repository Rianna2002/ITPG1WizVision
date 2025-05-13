import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import mlflow

mlflow.set_experiment("AOP_Market_Trend_Analysis")

st.set_page_config(page_title="AOP Market Trend Analysis Tool", layout="wide")

# Load label encoders used during training
town_classes = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON',
                'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
le_town = LabelEncoder()
le_town.fit(town_classes)

flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
le_flat_type = LabelEncoder()
le_flat_type.fit(flat_types)

storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21',
                 '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42',
                 '43 TO 45', '46 TO 48', '49 TO 51']
le_storey_range = LabelEncoder()
le_storey_range.fit(storey_ranges)

flat_models = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED', 'MAISONETTE',
               'APARTMENT', 'MODEL A-MAISONETTE', 'TERRACE', '2-ROOM', 'DBSS', 'IMPROVED-MAISONETTE',
               'MODEL A2', 'TYPE S1', 'TYPE S2', 'PREMIUM APARTMENT', 'MULTI GENERATION']
le_flat_model = LabelEncoder()
le_flat_model.fit(flat_models)

# Load trained XGBoost model
model = xgb.XGBRegressor()
model.load_model("xgboost_market_trend_analysis.json")

# Load dataset for trend analysis
@st.cache_data
def load_full_data():
    df1 = pd.read_csv('Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv')
    df2 = pd.read_csv('Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv')
    df3 = pd.read_csv('Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv')
    df4 = pd.read_csv('Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv')
    df5 = pd.read_csv('Resale flat prices based on registration date from Jan-2017 onwards.csv')

    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['town'] = df['town'].str.upper()
    return df

df_full = load_full_data()

# Layout Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 Price Trends", "📊 Town Comparison", "🔍 Predict Resale Price", 
    "🔄 Monthly Price Volatility", "🧱 Flat Type Price Trends", 
    "🗺️ Regional Price Heatmap", "⏳ Town-Level Price Appreciation (CAGR)", 
    "📊 Annual Transaction Volume", "🤖 Model Comparison"])

with tab1:
    st.header("📈 Resale Price Trends Over the Years")
    st.markdown("_This section shows the average resale flat prices over the years across Singapore. Users can optionally filter by town to observe how prices have changed in specific locations._")
    yearly_avg = df_full.groupby('year')['resale_price'].mean().reset_index()
    st.line_chart(yearly_avg.set_index('year'))

    selected_town = st.selectbox("Filter by Town (optional)", ["All"] + town_classes)
    if selected_town != "All":
        trend = df_full[df_full['town'] == selected_town].groupby('year')['resale_price'].mean().reset_index()
        st.line_chart(trend.set_index('year'))

with tab2:
    st.header("🏙️ Town-Level Price Distribution")
    st.markdown("_Displays the top 10 towns with the highest average resale prices. This helps users identify premium locations and understand geographic price differences._")
    town_avg = df_full.groupby('town')['resale_price'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(town_avg)

with tab3:
    st.header("🔍 Predict HDB Resale Price")
    st.markdown("_Use this tool to estimate the resale price of an HDB flat based on custom inputs such as town, flat type, floor area, and lease details using a pre-trained XGBoost model._")
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", town_classes)
        flat_type = st.selectbox("Flat Type", flat_types)
        storey_range = st.selectbox("Storey Range", storey_ranges)
        flat_model = st.selectbox("Flat Model", flat_models)

    with col2:
        floor_area = st.number_input("Floor Area (sqm)", min_value=30, max_value=200, value=80)
        lease_commence = st.number_input("Lease Commence Year", min_value=1960, max_value=2023, value=2000)
        remaining_lease = st.number_input("Remaining Lease (Years)", min_value=1, max_value=99, value=78)
        year = st.selectbox("Transaction Year", list(range(1990, 2024)), index=33)
        month_num = st.selectbox("Transaction Month", list(range(1, 13)), index=0)

    quarter = ((month_num - 1) // 3) + 1

    encoded_input = pd.DataFrame([{
        'town': le_town.transform([town])[0],
        'flat_type': le_flat_type.transform([flat_type])[0],
        'street_name': 0,
        'storey_range': le_storey_range.transform([storey_range])[0],
        'floor_area_sqm': floor_area,
        'flat_model': le_flat_model.transform([flat_model])[0],
        'lease_commence_date': lease_commence,
        'remaining_lease': remaining_lease,
        'year': year,
        'month_num': month_num,
        'quarter': quarter
    }])

    if st.button("💡 Predict Resale Price"):
        predicted_price = model.predict(encoded_input)[0]
        st.success(f"💰 Predicted Resale Price: SGD ${predicted_price:,.2f}")

        with mlflow.start_run():
            mlflow.log_params({
                "town": town,
                "flat_type": flat_type,
                "storey_range": storey_range,
                "floor_area_sqm": floor_area,
                "flat_model": flat_model,
                "lease_commence_year": lease_commence,
                "remaining_lease": remaining_lease,
                "year": year,
                "month": month_num
            })
            mlflow.log_metric("predicted_price", predicted_price)


with tab4:
    st.header("🔄 Monthly Price Volatility")
    st.markdown("_Visualizes the monthly standard deviation in resale prices, which reflects market volatility and how stable or fluctuating the prices are over time._")
    df_full['year_month'] = df_full['month'].dt.to_period('M').astype(str)
    monthly_std = df_full.groupby('year_month')['resale_price'].std().reset_index()
    st.line_chart(monthly_std.set_index('year_month'))

with tab5:
    st.header("🧱 Flat Type Price Trends")
    st.markdown("_Tracks the average resale prices by flat type (e.g., 3-room, 4-room) across the years. Useful for comparing affordability and value trends across flat types._")
    flat_type_trend = df_full.groupby(['year', 'flat_type'])['resale_price'].mean().reset_index()
    flat_type_trend_pivot = flat_type_trend.pivot(index='year', columns='flat_type', values='resale_price')
    st.line_chart(flat_type_trend_pivot)

with tab6:
    st.header("🗺️ Regional Price Heatmap by Town & Year")
    st.markdown("_Provides a heatmap of average resale prices by town and year, allowing for easy visual comparison of pricing trends across regions and time._")
    heatmap_data = df_full.groupby(['year', 'town'])['resale_price'].mean().unstack().fillna(0)
    st.dataframe(heatmap_data.style.background_gradient(cmap='YlGnBu', axis=1))

with tab7:
    st.header("⏳ Town-Level Price Appreciation (CAGR)")
    st.markdown("_Calculates the Compound Annual Growth Rate (CAGR) of resale prices from a selected base year to the latest year. It measures how much property prices have grown year-over-year in each town._")
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

with tab8:
    st.header("📊 Annual Transaction Volume")
    st.markdown("_Shows the total number of resale transactions per year and allows analysis of volume trends by town. This gives insights into market activity and flat supply._")
    volume_by_year = df_full.groupby('year').size().reset_index(name='Number of Transactions')
    st.bar_chart(volume_by_year.set_index('year'))

    st.subheader("🔍 Transactions by Town")
    selected_town_for_volume = st.selectbox("Select Town for Volume Analysis", sorted(df_full['town'].unique()))
    town_volume = df_full[df_full['town'] == selected_town_for_volume].groupby('year').size().reset_index(name='Transactions')
    st.line_chart(town_volume.set_index('year'))

with tab9:
    st.header("🤖 Model Comparison Across Algorithms")
    st.markdown("_This section compares three different predictive models (XGBoost, LSTM, ARIMA) based on error metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). It helps evaluate which model is more accurate and consistent for forecasting resale flat prices._")

    # Create a DataFrame for the comparison table
    model_comparison = pd.DataFrame({
        "Model": ["XGBoost", "LSTM", "ARIMA"],
        "MAE": [13241.14, 16454.58, 69533.19],
        "RMSE": [19265.25, 18599.16, 19361.81],
        "MAPE": ["4.76%", "3.07%", "12.59%"]
    })

    st.subheader("📋 Model Evaluation Metrics")
    st.table(model_comparison)

    # Display visual comparisons
    st.subheader("📷 Actual vs Predicted Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("XGBoost_AVP.png", caption="XGBoost Actual vs Predicted", use_container_width=True)

    with col2:
        st.image("LSTM_AVP.png", caption="LSTM Actual vs Predicted", use_container_width=True)

    with col3:
        st.image("ARIMA_AVP.png", caption="ARIMA Actual vs Predicted", use_container_width=True)
