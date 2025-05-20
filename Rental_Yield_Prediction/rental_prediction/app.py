import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor

# Set page configuration
st.set_page_config(
    page_title="Singapore Rental Prediction",
    page_icon="🏢",
    layout="wide"
)

# Application title
st.title("Singapore Property Rental Prediction")
st.markdown("## Predict monthly rent based on property details")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Predict", "Train Model", "Model Performance"])

# Add reset models button
if st.sidebar.button("🔄 Reset Models", key="reset_models"):
    # Clear all model-related session state
    st.session_state.predictors = {}
    st.session_state.model_trained = False
    st.session_state.model_loaded = False
    st.session_state.data_processor = None
    st.success("Models reset! Please retrain or reload your models.")
    st.experimental_rerun()

# Initialize session state variables if they don't exist
if 'predictors' not in st.session_state:
    st.session_state.predictors = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train model without MLflow
def train_model(data_processor, processed_data, model_type, tune_hyperparams=False):
    """
    Train model using the appropriate data for each model type
    """
    trainer = ModelTrainer()
    
    st.write(f"[DEBUG] Training {model_type.upper()} model...")
    
    try:
        if model_type == "xgboost":
            # XGBoost uses non-temporal data
            X_train, X_test, y_train, y_test = processed_data['xgboost']
            st.write(f"[DEBUG] XGBoost training data shape: {X_train.shape}")
            st.write(f"[DEBUG] XGBoost features: Categorical only (no temporal)")
            
            model = trainer.train_xgboost(processed_data, tune_hyperparams=tune_hyperparams)
            
            # Create predictor for testing
            predictor = RentalPredictor(
                preprocessor=data_processor,
                model=model,
                model_type="xgboost"
            )
            
            # Save the model explicitly
            st.write("[DEBUG] Saving XGBoost model...")
            model_path, preprocessor_path = predictor.save_model()
            st.write(f"[DEBUG] XGBoost model saved to: {model_path}")
            st.write(f"[DEBUG] Preprocessor saved to: {preprocessor_path}")
            
            # Verify the files were created
            if os.path.exists(model_path):
                st.success(f"✅ XGBoost model file verified: {model_path}")
            else:
                st.error(f"❌ XGBoost model file not found: {model_path}")
                
            # Test the model with different inputs
            st.write("[DEBUG] Testing XGBoost with different towns...")
            test_cases = [
                {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
                {'town': 'BEDOK', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
                {'town': 'CENTRAL', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
            ]
            
            st.write("[DEBUG] XGBoost Test Results:")
            for i, test_case in enumerate(test_cases):
                try:
                    prediction = predictor.predict(test_case)
                    st.write(f"  {test_case['town']}: S${prediction:.2f}")
                except Exception as e:
                    st.error(f"  {test_case['town']}: Error - {e}")
            
        elif model_type == "lstm":
            # LSTM uses temporal data
            X_train, X_test, y_train, y_test = processed_data['temporal']
            st.write(f"[DEBUG] LSTM training data shape: {X_train.shape}")
            st.write(f"[DEBUG] LSTM features: Categorical + temporal")
            
            model = trainer.train_lstm(processed_data)
            
            # Create predictor and save
            predictor = RentalPredictor(
                preprocessor=data_processor,
                model=model,
                model_type="lstm"
            )
            
            # Save the model explicitly
            st.write("[DEBUG] Saving LSTM model...")
            model_path, preprocessor_path = predictor.save_model()
            st.write(f"[DEBUG] LSTM model saved to: {model_path}")
            
            # Verify the files were created
            if os.path.exists(model_path):
                st.success(f"✅ LSTM model file verified: {model_path}")
            else:
                st.error(f"❌ LSTM model file not found: {model_path}")
            
        elif model_type == "arima":
            # ARIMA uses temporal data
            X_train, X_test, y_train, y_test = processed_data['temporal']
            st.write(f"[DEBUG] ARIMA training data shape: {X_train.shape}")
            st.write(f"[DEBUG] ARIMA features: Temporal focus")
            
            model = trainer.train_arima(processed_data)
            
            # Create predictor and save
            predictor = RentalPredictor(
                preprocessor=data_processor,
                model=model,
                model_type="arima"
            )
            
            # Save the model explicitly
            st.write("[DEBUG] Saving ARIMA model...")
            model_path, preprocessor_path = predictor.save_model()
            st.write(f"[DEBUG] ARIMA model saved to: {model_path}")
            
            # Verify the files were created
            if os.path.exists(model_path):
                st.success(f"✅ ARIMA model file verified: {model_path}")
            else:
                st.error(f"❌ ARIMA model file not found: {model_path}")
            
        else:
            st.error("Unsupported model type")
            return None
        
        # Check models directory
        st.write("[DEBUG] Checking models directory...")
        models_dir = "models"
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            st.write(f"[DEBUG] Files in models directory: {files}")
            
            # Show file sizes
            for file in files:
                file_path = os.path.join(models_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    st.write(f"  {file}: {size} bytes")
        else:
            st.error(f"❌ Models directory not found: {models_dir}")
        
        return predictor
        
    except Exception as e:
        st.error(f"Error during {model_type} training: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Function to make prediction with multiple models
def predict_rent_multi(input_features):
    predictions = {}
    
    for model_type, predictor in st.session_state.predictors.items():
        if predictor:
            try:
                predictions[model_type] = predictor.predict(input_features)
            except Exception as e:
                st.error(f"Error with {model_type} prediction: {e}")
                predictions[model_type] = None
    
    return predictions

# Prediction Page
if page == "Predict":
    st.header("Predict Monthly Rent")
    
    # Check if at least one model is loaded
    if not st.session_state.model_loaded and not st.session_state.model_trained:
        st.warning("No models are loaded. Please go to 'Train Model' page to train new models or load existing ones.")
    else:
        # Input form
        with st.form("prediction_form"):
            # Get available towns and flat types from the data
            data_path = "data/RentingOutofFlats2025.csv"
            if os.path.exists(data_path):
                data = load_data(data_path)
                towns = sorted(data['town'].unique())
                flat_types = sorted(data['flat_type'].unique())
                
                # Debug: Print actual flat types in the data
                print(f"[DEBUG] Actual flat types in data: {flat_types}")
            else:
                towns = ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", "BUKIT PANJANG", 
                         "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG", "CLEMENTI", "GEYLANG", "HOUGANG", 
                         "JURONG EAST", "JURONG WEST", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", 
                         "PUNGGOL", "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES", 
                         "TOA PAYOH", "WOODLANDS", "YISHUN"]
                flat_types = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                town = st.selectbox("Town", towns)
                flat_type = st.selectbox("Flat Type", flat_types)
            
            with col2:
                block = st.text_input("Block Number", "123")
                street_name = st.text_input("Street Name", "SAMPLE STREET")
            
            submit_button = st.form_submit_button("Predict Rent")
            
            if submit_button:
                # Create input features
                input_features = {
                    'town': town,
                    'block': block,
                    'street_name': street_name,
                    'flat_type': flat_type
                }
                
                # Make predictions using all available models
                predictions = predict_rent_multi(input_features)
                
                if predictions:
                    # Display results
                    st.subheader("Predicted Monthly Rent from All Models")
                    
                    # Create columns for each model prediction
                    model_cols = st.columns(len(predictions))
                    
                    # Define model colors for consistency
                    model_colors = {
                        "xgboost": "#1f77b4",  # Blue
                        "lstm": "#2ca02c",     # Green
                        "arima": "#d62728"     # Red
                    }
                    
                    # Display predictions
                    for i, (model_type, prediction) in enumerate(predictions.items()):
                        if prediction is not None:
                            with model_cols[i]:
                                st.metric(
                                    f"{model_type.upper()} Prediction", 
                                    f"S${prediction:.2f}",
                                    delta=None
                                )
                    
                    # Create a bar chart comparing model predictions
                    fig = go.Figure()
                    
                    for model_type, prediction in predictions.items():
                        if prediction is not None:
                            fig.add_trace(go.Bar(
                                x=[model_type.upper()],
                                y=[prediction],
                                name=model_type.upper(),
                                marker_color=model_colors.get(model_type.lower(), "gray")
                            ))
                    
                    fig.update_layout(
                        title="Model Predictions Comparison",
                        xaxis_title="Model",
                        yaxis_title="Predicted Rent (S$)",
                        height=400
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Additional analysis
                    st.subheader("Rental Analysis")
                    
                    # Calculate average prediction across models
                    valid_predictions = [p for p in predictions.values() if p is not None]
                    if valid_predictions:
                        avg_prediction = sum(valid_predictions) / len(valid_predictions)
                        
                        # Calculate annual rent based on average prediction
                        annual_rent = avg_prediction * 12
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Average Monthly Rent (All Models)", f"S${avg_prediction:.2f}")
                            st.metric("Annual Rent (Based on Average)", f"S${annual_rent:.2f}")
                        
                        with col2:
                            # Compare with average rent for that town and flat type
                            if os.path.exists(data_path):
                                avg_rent = data[(data['town'] == town) & (data['flat_type'] == flat_type)]['monthly_rent'].mean()
                                diff = avg_prediction - avg_rent
                                st.metric(f"Average Rent for {flat_type} in {town}", f"S${avg_rent:.2f}", f"{diff:+.2f}")
                                
                                # Comparison with similar properties
                                st.markdown("### Similar Properties")
                                similar = data[(data['town'] == town) & (data['flat_type'] == flat_type)].head(5)
                                st.dataframe(similar[['block', 'street_name', 'monthly_rent']])
                else:
                    st.error("No predictions available. Please ensure at least one model is trained or loaded.")

# Train Model Page
elif page == "Train Model":
    st.header("Train or Load Models")
    
    tab1, tab2 = st.tabs(["Train New Models", "Load Existing Models"])
    
    with tab1:
        st.subheader("Train New Rental Prediction Models")
        
        # Data loading section
        data_path = "data/RentingOutofFlats2025.csv"
        
        if os.path.exists(data_path):
            st.success(f"Found existing dataset: {data_path}")
            use_existing = st.checkbox("Use existing dataset", value=True)
            
            if use_existing:
                data = load_data(data_path)
            else:
                uploaded_file = st.file_uploader("Upload rental data CSV file", type=["csv"])
                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file)
                else:
                    data = None
        else:
            st.warning("No existing dataset found in the data directory.")
            uploaded_file = st.file_uploader("Upload rental data CSV file", type=["csv"])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
            else:
                data = None
        
        if data is not None:
            st.write(f"Dataset shape: {data.shape}")
            st.dataframe(data.head())
            
            # Model selection
            model_types = st.multiselect(
                "Select Models to Train", 
                ["xgboost", "lstm", "arima"],
                default=["xgboost"],
                help="XGBoost uses categorical features only, LSTM and ARIMA use temporal features"
            )
            
            # Add information about feature usage
            st.info("""
            **Feature Usage by Model:**
            - **XGBoost**: Uses categorical features (town, flat_type, block, street_name) only
            - **LSTM**: Uses all features including temporal (month, year)
            - **ARIMA**: Uses temporal features primarily
            """)
            
            # Hyperparameter tuning option
            tune_hyperparams = st.checkbox("Tune Hyperparameters (XGBoost only)", value=False)
            
            if st.button("Train Selected Models"):
                if not model_types:
                    st.error("Please select at least one model type to train.")
                else:
                    # Initialize data processor
                    data_processor = DataProcessor()
                    
                    # Preprocess data once for all models
                    with st.spinner("Preprocessing data for all models..."):
                        processed_data = data_processor.preprocess_data(data)
                    
                    # Store data processor in session state
                    st.session_state.data_processor = data_processor
                    
                    # Train each selected model
                    for model_type in model_types:
                        with st.spinner(f"Training {model_type.upper()} model... This may take a while."):
                            # Train model with appropriate data
                            predictor = train_model(
                                data_processor, processed_data, model_type, 
                                tune_hyperparams if model_type == "xgboost" else False
                            )
                            
                            if predictor:
                                # Store predictor in session state
                                st.session_state.predictors[model_type] = predictor
                                st.session_state.model_trained = True
                                st.session_state.model_loaded = True
                                
                                st.success(f"{model_type.upper()} model trained successfully!")
                    
                    st.success("All selected models trained! Go to the 'Predict' page to make predictions.")
    
    with tab2:
        st.subheader("Load Existing Models")
        
        # Get available models
        trainer = ModelTrainer()
        available_models = trainer.get_available_models()
        
        if available_models:
            st.write("Available Models:")
            
            # Display available models
            for model in available_models:
                with st.expander(f"{model['type'].upper()} Model"):
                    info = model['info']
                    st.write(f"**Type:** {info.get('model_type', 'Unknown')}")
                    if 'metrics' in info:
                        metrics = info['metrics']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                            st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                        with col2:
                            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                            st.metric("R²", f"{metrics.get('r2', 0):.2f}")
                    if 'created_at' in info:
                        st.write(f"**Created:** {info['created_at']}")
            
            # Select models to load
            models_to_load = st.multiselect(
                "Select Models to Load",
                [m['type'] for m in available_models]
            )
            
            if models_to_load and st.button("Load Selected Models"):
                # Check if preprocessor exists
                preprocessor_path = os.path.join("models", "data_processor.pkl")
                
                if os.path.exists(preprocessor_path):
                    # Load preprocessor
                    data_processor = DataProcessor()
                    predictor = RentalPredictor()
                    success = predictor.load_preprocessor(preprocessor_path)
                    
                    if success:
                        st.session_state.data_processor = predictor.preprocessor
                        
                        # Load each selected model
                        for model_type in models_to_load:
                            with st.spinner(f"Loading {model_type.upper()} model..."):
                                new_predictor = RentalPredictor(
                                    preprocessor=predictor.preprocessor
                                )
                                
                                success = new_predictor.load_from_file(model_type)
                                
                                if success:
                                    st.session_state.predictors[model_type] = new_predictor
                                    st.session_state.model_loaded = True
                                    st.success(f"{model_type.upper()} model loaded successfully!")
                                else:
                                    st.error(f"Failed to load {model_type.upper()} model.")
                        
                        if st.session_state.model_loaded:
                            st.success("All selected models loaded! Go to the 'Predict' page to make predictions.")
                    else:
                        st.error("Failed to load preprocessor.")
                else:
                    st.error("Preprocessor not found. Please train models first to create a preprocessor.")
        else:
            st.warning("No trained models found. Please train models first.")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Check for metrics log
    metrics_file = os.path.join("models", "metrics_log.json")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_log = json.load(f)
        
        if metrics_log:
            st.subheader("Model Performance History")
            
            # Convert to DataFrame for easier manipulation
            df_metrics = pd.DataFrame(metrics_log)
            
            # Add simplified timestamp
            df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])
            df_metrics['date'] = df_metrics['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Display metrics table
            st.dataframe(
                df_metrics[['model_type', 'date', 'metrics']].to_dict('records'),
                use_container_width=True
            )
            
            # Plot metrics comparison
            st.subheader("Metrics Comparison")
            
            # Extract metrics for plotting
            plot_data = []
            for entry in metrics_log:
                metrics = entry['metrics']
                plot_data.append({
                    'Model': entry['model_type'].upper(),
                    'RMSE': metrics.get('rmse', 0),
                    'MAE': metrics.get('mae', 0),
                    'MAPE': metrics.get('mape', 0),
                    'R²': metrics.get('r2', 0)
                })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Use tabs for different metric visualizations
            metric_tabs = st.tabs(["RMSE", "MAE", "MAPE", "R²"])
            
            with metric_tabs[0]:  # RMSE
                fig_rmse = px.bar(
                    plot_df, 
                    x='Model', 
                    y='RMSE',
                    title='Root Mean Squared Error (RMSE) Comparison',
                    color='Model'
                )
                st.plotly_chart(fig_rmse)
            
            with metric_tabs[1]:  # MAE
                fig_mae = px.bar(
                    plot_df, 
                    x='Model', 
                    y='MAE',
                    title='Mean Absolute Error (MAE) Comparison',
                    color='Model'
                )
                st.plotly_chart(fig_mae)
            
            with metric_tabs[2]:  # MAPE
                fig_mape = px.bar(
                    plot_df, 
                    x='Model', 
                    y='MAPE',
                    title='Mean Absolute Percentage Error (MAPE) Comparison',
                    color='Model'
                )
                st.plotly_chart(fig_mape)
            
            with metric_tabs[3]:  # R²
                fig_r2 = px.bar(
                    plot_df, 
                    x='Model', 
                    y='R²',
                    title='R-squared (R²) Comparison',
                    color='Model'
                )
                st.plotly_chart(fig_r2)
            
            # Model details
            st.subheader("Model Details")
            
            # Select model for details
            selected_model = st.selectbox(
                "Select Model for Details",
                plot_df['Model'].tolist()
            )
            
            # Find the selected model's entry
            selected_entry = next(
                (entry for entry in metrics_log if entry['model_type'].upper() == selected_model),
                None
            )
            
            if selected_entry:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Parameters")
                    params = selected_entry.get('params', {})
                    for param, value in params.items():
                        st.text(f"{param}: {value}")
                
                with col2:
                    st.markdown("### Metrics")
                    metrics = selected_entry['metrics']
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                    st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                    st.metric("R²", f"{metrics.get('r2', 0):.2f}")
                
                # Display model artifacts if available
                st.markdown("### Model Artifacts")
                model_type = selected_entry['model_type']
                
                # Check for visualizations
                artifacts = {
                    'Feature Importance': f"models/{model_type}_feature_importance.png",
                    'Training History': f"models/{model_type}_training_history.png",
                    'Predictions': f"models/{model_type}_predictions.png"
                }
                
                for artifact_name, artifact_path in artifacts.items():
                    if os.path.exists(artifact_path):
                        st.image(artifact_path, caption=artifact_name)
        else:
            st.warning("No metrics history found.")
    else:
        st.warning("No metrics log found. Train models to see performance metrics.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application predicts rental prices for properties in Singapore. "
    "It uses multiple machine learning models (XGBoost, LSTM, ARIMA) trained on historical rental data."
)
st.sidebar.markdown("### Instructions")
st.sidebar.info(
    "1. Train models or load existing ones from the 'Train Model' page\n"
    "2. Go to the 'Predict' page to make rental predictions with all models\n"
    "3. Check model performance metrics on the 'Model Performance' page"
)