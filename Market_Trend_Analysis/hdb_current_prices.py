'''This code is to create a Streamlit UI to query current HDB resale price averages per existing combination of

Month: 2025-04, 2025-05, 2025-06
Town: ANG MO KIO, BEDOK, BISHAN, BUKIT BATOK, BUKIT MERAH, BUKIT PANJANG, BUKIT TIMAH, CENTRAL AREA, CHOA CHU KANG, CLEMENTI, GEYLANG, HOUGANG, JURONG EAST, JURONG WEST, KALLANG/WHAMPOA, MARINE PARRADE, PASIR RIS, PUNGGOL, QUEENSTOWN, SENGKANG, SERANGOON, TAMPINES, TOA PAYOH, WOODLANDS, YISHUN
Flat Type: 1 ROOM, 2 ROOM, 3 ROOM, 4 ROOM, 5 ROOM, EXECUTIVE, MULTI GENERATION
Storey Range: 01 TO 03, 04 TO 06, 07 TO 09, 10 TO 12, 13 TO 15, 16 TO 18, 19 TO 21, 22 TO 24, 25 TO 27, 28 TO 30, 31 TO 33, 34 TO 36, 37 TO 39, 40 TO 42, 43 TO 45
Floor Area Range: 21-40, 41-60, 61-80, 81-100, 101-120, 121-140, 141-160
Flat Model: 2-room, 3Gen, Adjoined Flat, Apartment, DBSS, Improved, Improved-Maisonette, Maisonette, Model A, Model A-Maisonette, Model A2, Multi Generation, New Generation, Simplified, Premium Apartment, Premium Apartment Loft, Standard, Terrace, Type S1, Type S2
'''

import streamlit as st
import pandas as pd

# Load the new filtered full dataset
file_path = 'hdb_resale_filtered_2025_04_to_06.csv'
df = pd.read_csv(file_path)

# Extract year and month_num columns from 'month'
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
df['year'] = df['month'].dt.year
df['month_num'] = df['month'].dt.month
df['month'] = df['month'].dt.strftime('%Y-%m')  # Optional: keep month as original string format

st.title("HDB Resale Price Query (Apr-Jun 2025)")

# Start with full dataframe
filtered_df = df.copy()

# STEP 1: Month dropdown
month_options = sorted(filtered_df['month'].unique())
selected_month = st.selectbox("Select Month", [''] + month_options, format_func=lambda x: 'Select Month' if x == '' else x)

if selected_month != '':
    filtered_df = filtered_df[filtered_df['month'] == selected_month]

    # STEP 2: Town dropdown
    town_options = sorted(filtered_df['town'].unique())
    selected_town = st.selectbox("Select Town", [''] + town_options, format_func=lambda x: 'Select Town' if x == '' else x)

    if selected_town != '':
        filtered_df = filtered_df[filtered_df['town'] == selected_town]

        # STEP 3: Flat Type dropdown
        flat_type_options = sorted(filtered_df['flat_type'].unique())
        selected_flat_type = st.selectbox("Select Flat Type", [''] + flat_type_options, format_func=lambda x: 'Select Flat Type' if x == '' else x)

        if selected_flat_type != '':
            filtered_df = filtered_df[filtered_df['flat_type'] == selected_flat_type]

            # STEP 4: Storey Range dropdown
            storey_range_options = sorted(filtered_df['storey_range'].unique())
            selected_storey_range = st.selectbox("Select Storey Range", [''] + storey_range_options, format_func=lambda x: 'Select Storey Range' if x == '' else x)

            if selected_storey_range != '':
                filtered_df = filtered_df[filtered_df['storey_range'] == selected_storey_range]

                # STEP 5: Flat Model dropdown
                flat_model_options = sorted(filtered_df['flat_model'].unique())
                selected_flat_model = st.selectbox("Select Flat Model", [''] + flat_model_options, format_func=lambda x: 'Select Flat Model' if x == '' else x)

                if selected_flat_model != '':
                    filtered_df = filtered_df[filtered_df['flat_model'] == selected_flat_model]

                    # STEP 6: Floor Area input
                    floor_area_options = sorted(filtered_df['floor_area_sqm'].unique())
                    floor_area_min = int(min(floor_area_options))
                    floor_area_max = int(max(floor_area_options))
                    floor_area_default = floor_area_min
                    floor_area = st.number_input("Floor Area (sqm)", min_value=floor_area_min, max_value=floor_area_max, value=floor_area_default)

                    filtered_df = filtered_df[filtered_df['floor_area_sqm'] == floor_area]

                    # STEP 7: Lease Commence Year input
                    lease_year_options = sorted(filtered_df['lease_commence_date'].unique())
                    lease_year_min = int(min(lease_year_options))
                    lease_year_max = int(max(lease_year_options))
                    lease_year_default = lease_year_min
                    lease_commence_year = st.number_input("Lease Commence Year", min_value=lease_year_min, max_value=lease_year_max, value=lease_year_default)

                    filtered_df = filtered_df[filtered_df['lease_commence_date'] == lease_commence_year]

                    # ✅ Final result
                    if not filtered_df.empty:
                        avg_resale_price = filtered_df['resale_price'].mean()
                        st.success(f"Average Resale Price: ${avg_resale_price:,.2f}")
                        st.dataframe(filtered_df)
                    else:
                        st.warning("None available")

