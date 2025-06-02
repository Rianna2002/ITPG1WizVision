import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.data_processor import DataProcessor
from src.predictor import RentalPredictor

# Set page configuration
st.set_page_config(
    page_title="Singapore Rental Prediction",
    page_icon="🏢",
    layout="wide"
)

# Function to load a model with enhanced error handling
def load_model():
    """Load LSTM model from disk and return the predictor with improved error handling"""
    model_type = "lstm"  # Always load LSTM
    try:
        st.info(f"Attempting to load {model_type.upper()} model...")
        # Check if model file exists
        model_file = os.path.join("models", f"{model_type}_model.keras")
        if not os.path.exists(model_file):
            st.error(f"Model file not found: {model_file}")
            return None

        # Check if preprocessor exists
        preprocessor_path = os.path.join("models", "data_processor.pkl")
        if not os.path.exists(preprocessor_path):
            st.error(f"Preprocessor not found: {preprocessor_path}")
            return None
            
        # Create predictor
        predictor = RentalPredictor()
        
        # Load preprocessor
        st.info("Loading preprocessor...")
        success_preprocessor = predictor.load_preprocessor(preprocessor_path)
        if not success_preprocessor:
            st.error(f"Failed to load preprocessor from {preprocessor_path}")
            return None
        
        # Load model with explicit path
        st.info(f"Loading model from {model_file}...")
        success_model = predictor.load_from_file(model_type)
        if not success_model:
            st.error(f"Failed to load {model_type} model")
            return None
            
        return predictor
            
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to make prediction for multiple combinations
def predict_rent_multiple(towns, flat_types):
    """Make rental predictions for multiple town and flat_type combinations"""
    results = []
    
    if 'lstm' not in st.session_state.predictors:
        return results
        
    predictor = st.session_state.predictors['lstm']
    
    # Use default values for block and street_name
    block = "123"
    street_name = "SAMPLE STREET"
    
    # Make prediction for each combination
    for town in towns:
        for flat_type in flat_types:
            try:
                input_features = {
                    'town': town,
                    'block': block,
                    'street_name': street_name,
                    'flat_type': flat_type
                }
                prediction = predictor.predict(input_features)
                results.append({
                    'town': town,
                    'flat_type': flat_type,
                    'prediction': prediction
                })
            except Exception as e:
                st.error(f"Error predicting for {town}, {flat_type}: {e}")
    
    return results

# Prediction Page
def predict_rent_time_series_multiple(towns, flat_types, months_range=12):
    """
    Make predictions for multiple town/flat_type combinations over time
    
    Args:
        towns: List of towns to predict for
        flat_types: List of flat types to predict for
        months_range: Number of months before and after (default 12)
    
    Returns:
        Dictionary with predictions for each combo across time
    """
    # Get current date as reference point
    from datetime import datetime, timedelta
    current_date = datetime.now()
    
    # Generate list of dates for prediction
    dates = []
    for i in range(-months_range, months_range + 1):
        # Calculate target date
        target_date = current_date + timedelta(days=i*30)  # Approximate month
        dates.append({
            'year': target_date.year,
            'month': target_date.month,
            'label': target_date.strftime('%Y-%m')
        })
    
    # Initialize results
    time_series_results = {
        'dates': [d['label'] for d in dates],
        'predictions': {}
    }
    
    # Use default values for block and street_name
    block = "123"
    street_name = "SAMPLE STREET"
    
    # Only make predictions if LSTM is loaded
    if 'lstm' in st.session_state.predictors:
        predictor = st.session_state.predictors['lstm']
        
        # For each combination
        for town in towns:
            for flat_type in flat_types:
                combo_key = f"{town} - {flat_type}"
                predictions = []
                
                for date in dates:
                    # Copy input features and add temporal information
                    date_input = {
                        'town': town,
                        'block': block,
                        'street_name': street_name,
                        'flat_type': flat_type,
                        'year': date['year'],
                        'month': date['month']
                    }
                    
                    try:
                        prediction = predictor.predict(date_input)
                        predictions.append(prediction)
                    except Exception as e:
                        st.error(f"Error with LSTM prediction for {combo_key}, {date['label']}: {e}")
                        predictions.append(None)
                
                time_series_results['predictions'][combo_key] = predictions
    
    return time_series_results

# Function to calculate PICP (Prediction Interval Coverage Probability)
def calculate_picp(y_true, y_pred_lower, y_pred_upper):
    """Calculate percentage of true values that fall within prediction interval"""
    within_interval = np.logical_and(y_true >= y_pred_lower, y_true <= y_pred_upper)
    return np.mean(within_interval) * 100

# Function to get model metrics
def get_model_metrics():
    """Get metrics for the LSTM model from metrics_log.json and lstm_info.json"""
    model_type = "lstm"
    metrics_sources = {}
    
    try:
        # First try metrics_log.json (training metrics)
        metrics_file = os.path.join("models", "metrics_log.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_log = json.load(f)
            
            # Find the metrics for the specified model
            for entry in metrics_log:
                if entry['model_type'] == model_type:
                    metrics_sources['metrics_log'] = entry.get('metrics', {})
        
        # Then try lstm_info.json (evaluation metrics)
        model_info_file = os.path.join("models", f"{model_type}_info.json")
        if os.path.exists(model_info_file):
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)
                metrics_sources['lstm_info'] = model_info.get('metrics', {})
        
        return metrics_sources
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
    
    return {}

# Function to get top features from model
def get_top_features(data, n_features=3):
    """Get top n features that influence the LSTM predictions using a surrogate model"""
    try:
        # Train a simple Random Forest as a surrogate model
        X = data.drop('monthly_rent', axis=1)
        # Drop temporal features to get base features
        if 'month' in X.columns:
            X = X.drop(['month', 'year'], axis=1)
            
        # Handle categorical features
        X_encoded = pd.get_dummies(X)
        
        # Train surrogate model
        surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
        surrogate_model.fit(X_encoded, data['monthly_rent'])
        
        # Get feature importances from surrogate model
        importances = surrogate_model.feature_importances_
        feature_names = X_encoded.columns
        
        # Sort and get top features
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:n_features]
        
        # Return top feature names and importances
        return [(feature_names[i], importances[i]) for i in top_indices]
    except Exception as e:
        st.error(f"Error getting top features: {e}")
    
    return []

# Initialize session state variables if they don't exist
if 'predictors' not in st.session_state:
    st.session_state.predictors = {}
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Predict"

# Application title
st.title("Singapore Property Rental Prediction (LSTM Model)")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Predict", "Model Statistics"],
    index=0 if st.session_state.current_page == "Predict" else 1
)
st.session_state.current_page = page

# Model status indicator in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")

# Auto-load LSTM model
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading LSTM model... Please wait..."):
        predictor = load_model()
        
        if predictor:
            # Store in session state
            st.session_state.predictors["lstm"] = predictor
            st.session_state.model_loaded = True
            st.sidebar.success("✅ LSTM model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load LSTM model.")

# Show model status in sidebar
if st.session_state.model_loaded:
    st.sidebar.success("✅ LSTM model loaded and ready")
else:
    st.sidebar.warning("⚠️ LSTM model not loaded")

# Footer in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application predicts rental prices for properties in Singapore "
    "using an LSTM deep learning model trained on historical rental data."
)

# Handle page selection
if page == "Predict":
    # Main prediction interface
    st.header("Predict Monthly Rent")
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.warning("LSTM model is not loaded. Please check the error messages and refresh the page.")
    else:
        # Input form
        with st.form("prediction_form"):
            # Get available towns and flat types from the data
            data_path = "data/RentingOutofFlats2025.csv"
            if os.path.exists(data_path):
                data = load_data(data_path)
                towns = sorted(data['town'].unique())
                flat_types = sorted(data['flat_type'].unique())
            else:
                towns = ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", 
                        "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", 
                        "JURONG EAST", "JURONG WEST", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", 
                        "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES", 
                        "TOA PAYOH", "WOODLANDS", "YISHUN"]
                flat_types = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
            
            # Multi-select for towns (limit to 3)
            selected_towns = st.multiselect(
                "Select Towns (up to 3)",
                towns,
                default=[towns[0]],
                max_selections=3
            )
            
            # Multi-select for flat types (limit to 3)
            selected_flat_types = st.multiselect(
                "Select Flat Types (up to 3)",
                flat_types,
                default=[flat_types[2]],  # Default to 3 ROOM
                max_selections=3
            )
            
            # Help text
            st.markdown("*Note: Block number and street name are fixed to standard values*")
            
            submit_button = st.form_submit_button("Predict Rent")
            
            if submit_button:
                # Validate selections
                if not selected_towns:
                    st.error("Please select at least one town")
                elif not selected_flat_types:
                    st.error("Please select at least one flat type")
                else:
                    # Make predictions for all combinations
                    with st.spinner("Calculating predictions..."):
                        results = predict_rent_multiple(selected_towns, selected_flat_types)
                    
                    if results:
                        # Display current predictions in a table
                        st.subheader("Current Monthly Rent Predictions")
                        
                        # Create DataFrame for display
                        results_df = pd.DataFrame(results)
                        
                        # Format the prediction column
                        results_df['prediction'] = results_df['prediction'].apply(lambda x: f"S${x:.2f}")
                        
                        # Reshape for better display - pivot table with towns as rows and flat types as columns
                        pivot_df = results_df.pivot(index='town', columns='flat_type', values='prediction')
                        
                        # Display the pivot table
                        st.dataframe(pivot_df, use_container_width=True)
                        
                        # Generate time series prediction automatically
                        st.subheader("Rental Price Forecast (±12 Months)")
                        
                        # Make time series predictions
                        with st.spinner("Generating time series forecast..."):
                            time_series_data = predict_rent_time_series_multiple(selected_towns, selected_flat_types)
                        
                        if time_series_data['predictions']:
                            # Create line chart for time series predictions
                            fig = go.Figure()
                            
                            reference_point = len(time_series_data['dates']) // 2
                            
                            # Create a color palette with enough distinct colors
                            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
                            
                            # Calculate min/max values for y-axis range
                            all_values = []
                            
                            # Create a DataFrame for tabular display
                            table_data = {"Date": time_series_data['dates']}
                            
                            # Add each combination as a line on the plot
                            color_index = 0
                            for combo_key, predictions in time_series_data['predictions'].items():
                                if any(p is not None for p in predictions):
                                    # Add to plot
                                    fig.add_trace(go.Scatter(
                                        x=time_series_data['dates'],
                                        y=predictions,
                                        mode='lines+markers',
                                        name=combo_key,
                                        line=dict(color=colors[color_index % len(colors)], width=2),
                                        hoverinfo='text',
                                        hovertext=[f"{combo_key}, {date}: S${pred:.2f}" for date, pred in 
                                                zip(time_series_data['dates'], predictions)]
                                    ))
                                    
                                    # Add to table data
                                    table_data[combo_key] = [f"S${pred:.2f}" if pred is not None else "N/A" 
                                                            for pred in predictions]
                                    
                                    # Add to values list for y-axis range calculation
                                    all_values.extend([v for v in predictions if v is not None])
                                    
                                    # Increment color index
                                    color_index += 1
                            
                            # Add a vertical line for current date
                            fig.add_shape(
                                type="line",
                                x0=time_series_data['dates'][reference_point],
                                y0=min(all_values) * 0.95 if all_values else 0,
                                x1=time_series_data['dates'][reference_point],
                                y1=max(all_values) * 1.05 if all_values else 0,
                                line=dict(color="black", width=1, dash="dash"),
                            )
                            
                            # Set y-axis range to not start from zero
                            y_min = min(all_values) * 0.95 if all_values else 0
                            y_max = max(all_values) * 1.05 if all_values else 0
                            
                            fig.update_layout(
                                title="Rental Price Forecast Over Time (LSTM Model)",
                                xaxis_title="Date",
                                yaxis_title="Predicted Rent (S$)",
                                hovermode="closest",
                                height=600,  # Increased height for better visibility
                                yaxis=dict(range=[y_min, y_max]),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="center",
                                    x=0.5
                                )
                            )
                            
                            # Add zone highlighting
                            # Past zone (light gray)
                            fig.add_shape(
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=time_series_data['dates'][0],
                                y0=y_min,
                                x1=time_series_data['dates'][reference_point-1],
                                y1=y_max,
                                fillcolor="lightgray",
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                            )

                            # Current zone (light green)
                            fig.add_shape(
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=time_series_data['dates'][reference_point],
                                y0=y_min,
                                x1=time_series_data['dates'][reference_point],
                                y1=y_max,
                                fillcolor="rgb(0, 255, 0)",
                                opacity=0.3,
                                layer="below",
                                line_width=0,
                            )

                            # Future zone (light blue)
                            fig.add_shape(
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=time_series_data['dates'][reference_point+1],
                                y0=y_min,
                                x1=time_series_data['dates'][-1],
                                y1=y_max,
                                fillcolor="lightblue",
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                            )

                            # Add annotations for zones
                            fig.add_annotation(
                                x=time_series_data['dates'][reference_point//2],
                                y=y_max*0.95,
                                text="Historical Data",
                                showarrow=False,
                                font=dict(size=12)
                            )

                            fig.add_annotation(
                                x=time_series_data['dates'][reference_point],
                                y=y_max*0.95,
                                text="Current",
                                showarrow=False,
                                font=dict(size=12)
                            )

                            fig.add_annotation(
                                x=time_series_data['dates'][reference_point + (len(time_series_data['dates'])-reference_point)//2],
                                y=y_max*0.95,
                                text="Forecast",
                                showarrow=False,
                                font=dict(size=12)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display tabular data
                            st.subheader("Monthly Forecast Values")
                            forecast_df = pd.DataFrame(table_data)
                            
                            # Highlight current month row
                            def highlight_current_month(row):
                                if row.name == reference_point:
                                    return ['background-color: #0A0'] * len(row)
                                return [''] * len(row)
                            
                            # Display the table with styling
                            st.dataframe(
                                forecast_df.style.apply(highlight_current_month, axis=1),
                                use_container_width=True
                            )
                            
                            # Add an explanation of the forecast
                            with st.expander("About this forecast"):
                                st.markdown("""
                                This forecast shows predicted rental prices for the selected properties over time (12 months 
                                before and after the current date). The LSTM model uses historical patterns to predict 
                                future rental trends and shows what the rental price likely was historically.
                                
                                * The vertical dashed line represents the current month
                                * Past predictions show what the rental price likely was historically (gray area)
                                * Future predictions show expected rental price trends (blue area)
                                * The current month is highlighted in green
                                * You can compare up to 9 different town/flat type combinations
                                """)
                        else:
                            st.warning("Unable to generate time series forecast. Please check if the LSTM model is loaded correctly.")

elif page == "Model Statistics":
    st.header("LSTM Model Performance Statistics")
    
    # Check if LSTM model is loaded
    if not st.session_state.model_loaded or 'lstm' not in st.session_state.predictors:
        st.warning("LSTM model is not loaded. Please go to the Predict page to load the model first.")
    else:
        # Load data for feature analysis
        data_path = "data/RentingOutofFlats2025.csv"
        if os.path.exists(data_path):
            data = load_data(data_path)
            
            # Get metrics for LSTM model from both sources
            metrics_sources = get_model_metrics()
            
            # Add tabs to show both sets of metrics
            tabs = st.tabs(["Evaluation Metrics", "Training Metrics", "Comparison"])
            
            # Evaluation metrics tab (from lstm_info.json)
            with tabs[0]:
                if 'lstm_info' in metrics_sources:
                    metrics = metrics_sources['lstm_info']
                    st.subheader("Evaluation Metrics (from lstm_info.json)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.2f}" if 'rmse' in metrics else "N/A")
                    with col2:
                        st.metric("MAPE", f"{metrics.get('mape', 'N/A'):.2f}%" if 'mape' in metrics else "N/A")
                    with col3:
                        # Use actual PICP from metrics instead of hardcoded value
                        picp_value = metrics.get('picp', 0.0)
                        st.metric("PICP", f"{picp_value*100:.1f}%" if isinstance(picp_value, (int, float)) else "N/A")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{metrics.get('r2', 'N/A'):.4f}" if 'r2' in metrics else "N/A")
                    with col2:
                        mpiw_value = metrics.get('mpiw', 0.0)
                        st.metric("MPIW", f"{mpiw_value:.2f}" if isinstance(mpiw_value, (int, float)) else "N/A")
                    
                    # Show feature importance if available
                    if 'feature_importance' in metrics:
                        st.subheader("Feature Importance")
                        feature_imp = metrics['feature_importance']
                        feature_df = pd.DataFrame({
                            'Feature': list(feature_imp.keys()),
                            'Importance': list(feature_imp.values())
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(feature_df, x='Feature', y='Importance', title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No evaluation metrics found in lstm_info.json")
                    
            # Training metrics tab (from metrics_log.json)
            with tabs[1]:
                if 'metrics_log' in metrics_sources:
                    metrics = metrics_sources['metrics_log']
                    st.subheader("Training Metrics (from metrics_log.json)")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.2f}" if 'rmse' in metrics else "N/A")
                    with col2:
                        st.metric("MAPE", f"{metrics.get('mape', 'N/A'):.2f}%" if 'mape' in metrics else "N/A")
                    with col3:
                        st.metric("MAE", f"{metrics.get('mae', 'N/A'):.2f}" if 'mae' in metrics else "N/A")
                    
                    st.metric("R² Score", f"{metrics.get('r2', 'N/A'):.4f}" if 'r2' in metrics else "N/A")
                    
                    if 'mse' in metrics:
                        st.metric("MSE", f"{metrics.get('mse', 'N/A'):.2f}")
                else:
                    st.warning("No training metrics found in metrics_log.json")
            
            # Comparison tab to explain differences
            with tabs[2]:
                st.subheader("Understanding the Metrics Differences")
                st.markdown("""
                ### Why are the metrics different?

                The two JSON files contain metrics calculated at different stages:

                1. **metrics_log.json**: Contains metrics calculated during model training, typically on validation data.
                
                2. **lstm_info.json**: Contains metrics from a more comprehensive evaluation, including:
                   - Prediction intervals (PICP, MPIW)
                   - Feature importance analysis
                   - Metrics on different test/evaluation data

                #### Key differences:
                - **RMSE**: 493.39 (training) vs 1082.76 (evaluation)
                - **MAPE**: 14.94% (training) vs 38.39% (evaluation)
                
                The evaluation metrics are likely more representative of real-world performance, as they were calculated on separate test data or using a different evaluation methodology.
                """)
                
                # Create comparison table
                if 'metrics_log' in metrics_sources and 'lstm_info' in metrics_sources:
                    training = metrics_sources['metrics_log']
                    evaluation = metrics_sources['lstm_info']
                    
                    compare_data = {
                        'Metric': ['RMSE', 'MAPE', 'R²'],
                        'Training Value': [
                            f"{training.get('rmse', 'N/A'):.2f}",
                            f"{training.get('mape', 'N/A'):.2f}%",
                            f"{training.get('r2', 'N/A'):.4f}"
                        ],
                        'Evaluation Value': [
                            f"{evaluation.get('rmse', 'N/A'):.2f}",
                            f"{evaluation.get('mape', 'N/A'):.2f}%",
                            f"{evaluation.get('r2', 'N/A'):.4f}"
                        ]
                    }
                    
                    st.subheader("Metrics Comparison")
                    st.table(pd.DataFrame(compare_data))