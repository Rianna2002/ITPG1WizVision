import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("models/propertypriceforecasting_xgboost_model.model")

# Load previous month's resale prices (generated in notebook)
prev_prices = pd.read_csv("data/prev_prices.csv")

# Function to get previous resale price
def get_prev_price(flat_type, year, month):
    row = prev_prices[(prev_prices['flat_type'] == flat_type) & 
                      (prev_prices['year'] == year) & 
                      (prev_prices['month_num'] == month - 1)]
    if not row.empty:
        return float(row['prev_month_resale_price'].values[0])
    fallback = prev_prices[(prev_prices['flat_type'] == flat_type) & 
                           ((prev_prices['year'] < year) | 
                           ((prev_prices['year'] == year) & (prev_prices['month_num'] < month - 1)))]
    if not fallback.empty:
        return float(fallback.sort_values(by=['year', 'month_num'], ascending=False)['prev_month_resale_price'].values[0])
    return np.nan

# Feature names from training data
feature_names = [
    'flat_type', 'block', 'storey_range', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease',
    'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG',
    'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG',
    'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS',
    'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES',
    'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN',
    'flat_model_2-ROOM', 'flat_model_3GEN', 'flat_model_ADJOINED FLAT', 'flat_model_APARTMENT', 'flat_model_DBSS',
    'flat_model_IMPROVED', 'flat_model_IMPROVED-MAISONETTE', 'flat_model_MAISONETTE', 'flat_model_MODEL A',
    'flat_model_MODEL A-MAISONETTE', 'flat_model_MODEL A2', 'flat_model_MULTI GENERATION', 'flat_model_NEW GENERATION',
    'flat_model_PREMIUM APARTMENT', 'flat_model_PREMIUM APARTMENT LOFT', 'flat_model_PREMIUM MAISONETTE',
    'flat_model_SIMPLIFIED', 'flat_model_STANDARD', 'flat_model_TERRACE', 'flat_model_TYPE S1', 'flat_model_TYPE S2',
    'year', 'month_num', 'prev_month_resale_price'
]

# Streamlit UI layout
st.title("HDB Resale Price Predictor")

# ROW 1 — Input + Metrics
col_left, col_right = st.columns([6, 2.5])

with col_left:
    st.write("Enter the details below to predict the resale price.")

    flat_type_order = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'MULTI-GENERATION', 'EXECUTIVE']
    storey_range_order = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21',
                          '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42',
                          '43 TO 45', '46 TO 48', '49 TO 51']
    town_order = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
                  'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                  'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG',
                  'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    flat_model_order = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED', 'PREMIUM APARTMENT',
                        'MAISONETTE', 'APARTMENT', 'MODEL A2', 'TYPE S1', 'TYPE S2', 'ADJOINED FLAT', 'TERRACE', 'DBSS',
                        'MODEL A-MAISONETTE', 'PREMIUM MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT LOFT',
                        'IMPROVED-MAISONETTE', '2-ROOM', '3GEN']

    flat_type = st.selectbox("Flat Type", flat_type_order)
    block = st.number_input("Block Number", min_value=1, max_value=999, value=120)
    storey_range = st.selectbox("Storey Range", storey_range_order)
    floor_area = st.number_input("Floor Area (sqm)", min_value=30, max_value=200, value=100)
    lease_commence = st.number_input("Lease Commence Year", min_value=1960, max_value=2023, value=1990)
    remaining_lease = st.number_input("Remaining Lease (months)", min_value=0, max_value=1188, value=900)
    year = st.number_input("Year of Transaction", min_value=2023, max_value=2025, value=2023)
    month_num = st.number_input("Month of Transaction (1-12)", min_value=1, max_value=12, value=1)

    type_encoded = flat_type_order.index(flat_type)
    prev_month_resale_price = get_prev_price(type_encoded, year, month_num)
    st.number_input("Previous Month's Avg Resale Price", min_value=100000, max_value=1000000, value=int(prev_month_resale_price))

    selected_town = st.selectbox("Select Town", town_order, key='town_select')
    town_encoded = [1 if town == selected_town else 0 for town in town_order]
    selected_flat_model = st.selectbox("Select Flat Model", flat_model_order, key='flat_model_select')
    flat_model_encoded = [1 if fm == selected_flat_model else 0 for fm in flat_model_order]

    features = np.array([
        type_encoded, block, storey_range_order.index(storey_range), floor_area, lease_commence, remaining_lease,
        *town_encoded, *flat_model_encoded, year, month_num, prev_month_resale_price
    ]).reshape(1, -1)

    dmatrix = xgb.DMatrix(features, feature_names=feature_names)

with col_right:
    st.markdown("### Model Evaluation Metrics")
    st.metric(label="MAPE", value="6.23%")
    st.metric(label="PICP (95%)", value="0.890")
    st.metric(label="MPIW (95%)", value="$196,237.75")

# ROW 2 — Charts (full-width after prediction)
if st.button("Predict Price"):
    predicted_prices = []
    months_str = []

    for y in range(2023, 2026):
        for m in range(1, 13):
            prev_price = get_prev_price(type_encoded, y, m)
            feature_vector = np.array([
                type_encoded, block, storey_range_order.index(storey_range), floor_area, lease_commence, remaining_lease,
                *town_encoded, *flat_model_encoded, y, m, prev_price
            ]).reshape(1, -1)
            dmatrix = xgb.DMatrix(feature_vector, feature_names=feature_names)
            predicted_price = xgb_model.predict(dmatrix)[0]
            predicted_prices.append(predicted_price)
            months_str.append(f"{pd.Timestamp(year=y, month=m, day=1):%b %Y}")

    try:
        idx = months_str.index(f"{pd.Timestamp(year=year, month=month_num, day=1):%b %Y}")
        st.success(f"Predicted Resale Price for {months_str[idx]}: **${predicted_prices[idx]:,.2f}**")
    except ValueError:
        pass

    st.markdown("---")
    st.subheader("Predicted HDB Resale Prices (2023–2025)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(months_str, predicted_prices, marker='o', color='green')
    y_min, y_max = min(predicted_prices), max(predicted_prices)
    pad = (y_max - y_min) * 0.65
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xticks(range(len(months_str)))
    ax.set_xticklabels(months_str, rotation=45, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Resale Price")
    ax.set_title("Predicted HDB Resale Prices (2023–2025)")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Feature Importances 📊")
    importances = xgb_model.get_score(importance_type="weight")
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_n = 9
    top_features = sorted_importances[:top_n]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([x[0] for x in top_features], [x[1] for x in top_features], color="skyblue")
    ax.set_xlabel("Feature Importance (Weight)")
    ax.set_title(f"Top {top_n} Features Used by XGBoost")
    ax.invert_yaxis()
    st.pyplot(fig)
