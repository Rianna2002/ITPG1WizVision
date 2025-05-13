import streamlit as st
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.xgboost
import mlflow.keras
import mlflow.sklearn
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
# Add this in the sidebar or main area
if st.button("🔄 Reset Models", key="reset_models"):
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

# Function to train model
# Replace the train_model function in app.py with this version that has more debugging:

def train_model(data_processor, processed_data, model_type, tune_hyperparams=False):
    """
    Train model using the appropriate data for each model type
    """
    trainer = ModelTrainer()
    
    st.write(f"[DEBUG] Training {model_type.upper()} model...")
    
    if model_type == "xgboost":
        # XGBoost uses non-temporal data
        X_train, X_test, y_train, y_test = processed_data['xgboost']
        st.write(f"[DEBUG] XGBoost training data shape: {X_train.shape}")
        st.write(f"[DEBUG] XGBoost features: Categorical only (no temporal)")
        
        model = trainer.train_xgboost(processed_data, tune_hyperparams=tune_hyperparams)
        
        # Test the model with different inputs
        st.write("[DEBUG] Testing XGBoost with different towns...")
        
        # Create test inputs
        test_cases = [
            {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
            {'town': 'BEDOK', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
            {'town': 'CENTRAL', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
        ]
        
        # Create predictor for testing
        predictor = RentalPredictor(
            preprocessor=data_processor,
            model=model,
            model_type="xgboost"
        )
        
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
        
    elif model_type == "arima":
        # ARIMA uses temporal data
        X_train, X_test, y_train, y_test = processed_data['temporal']
        st.write(f"[DEBUG] ARIMA training data shape: {X_train.shape}")
        st.write(f"[DEBUG] ARIMA features: Temporal focus")
        
        model = trainer.train_arima(processed_data)
        
    else:
        st.error("Unsupported model type")
        return None
    
    best_model, best_model_name = trainer.get_best_model()
    
    # Create predictor with appropriate model type
    predictor = RentalPredictor(
        preprocessor=data_processor,
        model=best_model,
        model_type=best_model_name
    )
    
    # Save model
    model_path, preprocessor_path = predictor.save_model()
    
    return predictor

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
        
        # MLflow or file-based model loading
        load_option = st.radio("Load models from:", ["MLflow", "Local File"])
        
        if load_option == "MLflow":
            # Get available runs from MLflow
            try:
                experiment_name = "rental_prediction"
                experiment = mlflow.get_experiment_by_name(experiment_name)
                
                if experiment:
                    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                    
                    if len(runs) > 0:
                        run_ids = runs['run_id'].tolist()
                        run_names = [f"Run {i+1}: {runs.iloc[i]['tags.mlflow.runName']} ({runs.iloc[i]['run_id']})" 
                                   for i in range(len(runs))]
                        
                        # Allow selecting multiple runs
                        selected_runs = st.multiselect("Select MLflow Runs", run_names)
                        
                        if selected_runs and st.button("Load Selected Models"):
                            # Initialize data processor
                            data_processor = DataProcessor()
                            
                            # Process training data if available
                            if os.path.exists(data_path):
                                data = load_data(data_path)
                                data_processor.preprocess_data(data)
                                st.session_state.data_processor = data_processor
                            
                            # Load each selected model
                            for selected_run in selected_runs:
                                selected_run_id = run_ids[run_names.index(selected_run)]
                                
                                # Determine model type from run name
                                run_name = runs[runs['run_id'] == selected_run_id]['tags.mlflow.runName'].iloc[0]
                                
                                if "XGBoost" in run_name:
                                    model_type = "xgboost"
                                elif "LSTM" in run_name:
                                    model_type = "lstm"
                                elif "ARIMA" in run_name:
                                    model_type = "arima"
                                else:
                                    # Let user select model type
                                    model_type = st.selectbox(
                                        f"Model Type for {run_name}", 
                                        ["xgboost", "lstm", "arima"]
                                    )
                                
                                with st.spinner(f"Loading {model_type.upper()} model from MLflow..."):
                                    predictor = RentalPredictor()
                                    success = predictor.load_from_mlflow(selected_run_id, model_type)
                                    
                                    if success:
                                        predictor.preprocessor = data_processor
                                        st.session_state.predictors[model_type] = predictor
                                        st.session_state.model_loaded = True
                                        
                                        st.success(f"{model_type.upper()} model loaded successfully!")
                            
                            if st.session_state.model_loaded:
                                st.success("All selected models loaded! Go to the 'Predict' page to make predictions.")
                    else:
                        st.warning("No MLflow runs found. Train models first.")
                else:
                    st.warning(f"Experiment '{experiment_name}' not found in MLflow.")
            except Exception as e:
                st.error(f"Error accessing MLflow: {e}")
                st.info("Make sure MLflow server is running and accessible.")
        
        else:  # Local File
            model_dir = "models"
            
            if os.path.exists(model_dir):
                # List available models
                model_files = [f for f in os.listdir(model_dir) 
                             if os.path.isfile(os.path.join(model_dir, f)) and not f.endswith('.pkl')]
                
                if len(model_files) > 0:
                    # Allow selecting multiple model files
                    selected_models = st.multiselect("Select Model Files", model_files)
                    
                    if selected_models and st.button("Load Selected Models"):
                        # Check for preprocessor
                        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
                        
                        if os.path.exists(preprocessor_path):
                            # Load each selected model
                            for selected_model in selected_models:
                                model_path = os.path.join(model_dir, selected_model)
                                
                                # Determine model type from filename
                                if "xgboost" in selected_model.lower():
                                    model_type = "xgboost"
                                elif "lstm" in selected_model.lower():
                                    model_type = "lstm"
                                elif "arima" in selected_model.lower():
                                    model_type = "arima"
                                else:
                                    # Let user select model type
                                    model_type = st.selectbox(
                                        f"Model Type for {selected_model}", 
                                        ["xgboost", "lstm", "arima"]
                                    )
                                
                                with st.spinner(f"Loading {model_type.upper()} model from file..."):
                                    predictor = RentalPredictor()
                                    success = predictor.load_from_file(model_path, model_type)
                                    
                                    if success:
                                        # Load preprocessor
                                        preprocessor_success = predictor.load_preprocessor(preprocessor_path)
                                        
                                        if preprocessor_success:
                                            st.session_state.predictors[model_type] = predictor
                                            st.session_state.model_loaded = True
                                            
                                            st.success(f"{model_type.upper()} model loaded successfully!")
                                        else:
                                            st.error("Failed to load preprocessor.")
                                    else:
                                        st.error(f"Failed to load {model_type.upper()} model from file.")
                            
                            if st.session_state.model_loaded:
                                st.success("All selected models loaded! Go to the 'Predict' page to make predictions.")
                        else:
                            st.warning("Preprocessor not found. You need both the model and preprocessor to make predictions.")
                else:
                    st.warning("No model files found in the models directory. Train models first.")
            else:
                st.warning("Models directory not found. Train models first.")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    if not st.session_state.model_trained and not st.session_state.model_loaded:
        st.warning("No models have been trained or loaded yet. Please go to the 'Train Model' page first.")
    else:
        # Check if MLflow is available
        try:
            experiment_name = "rental_prediction"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                
                if len(runs) > 0:
                    # Display runs and metrics
                    st.subheader("MLflow Runs and Metrics")
                    
                    # Include MAPE in the metrics display
                    metrics_df = runs[['tags.mlflow.runName', 'metrics.rmse', 'metrics.mae', 'metrics.mape', 'metrics.r2', 'tags.mlflow.source.name']]
                    metrics_df.columns = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'Source']
                    
                    st.dataframe(metrics_df)
                    
                    # Plot metrics comparison
                    st.subheader("Metrics Comparison")
                    
                    # Use tabs for different metric visualizations
                    metric_tabs = st.tabs(["RMSE", "MAE", "MAPE"])
                    
                    with metric_tabs[0]:  # RMSE
                        fig_rmse = px.bar(
                            metrics_df, 
                            x='Model', 
                            y='RMSE',
                            title='Root Mean Squared Error (RMSE) Comparison',
                            color='Model'
                        )
                        st.plotly_chart(fig_rmse)
                    
                    with metric_tabs[1]:  # MAE
                        fig_mae = px.bar(
                            metrics_df, 
                            x='Model', 
                            y='MAE',
                            title='Mean Absolute Error (MAE) Comparison',
                            color='Model'
                        )
                        st.plotly_chart(fig_mae)
                    
                    with metric_tabs[2]:  # MAPE
                        fig_mape = px.bar(
                            metrics_df, 
                            x='Model', 
                            y='MAPE (%)',
                            title='Mean Absolute Percentage Error (MAPE) Comparison',
                            color='Model'
                        )
                        st.plotly_chart(fig_mape)
                    
                    # Select run for detailed view
                    selected_run_name = st.selectbox("Select Run for Details", metrics_df['Model'].tolist())
                    selected_run = runs[runs['tags.mlflow.runName'] == selected_run_name].iloc[0]
                    
                    st.subheader(f"Details for {selected_run_name}")
                    
                    # Display parameters
                    st.markdown("### Parameters")
                    params = {k: v for k, v in selected_run.items() if k.startswith('params.')}
                    params = {k.replace('params.', ''): v for k, v in params.items()}
                    
                    for param, value in params.items():
                        st.text(f"{param}: {value}")
                    
                    # Display artifacts if available
                    st.markdown("### Artifacts")
                    
                    run_id = selected_run['run_id']
                    client = mlflow.tracking.MlflowClient()
                    artifacts = client.list_artifacts(run_id)
                    
                    for artifact in artifacts:
                        if artifact.path.endswith('.png'):
                            # Download and display image
                            artifact_path = mlflow.artifacts.download_artifacts(
                                run_id=run_id,
                                artifact_path=artifact.path
                            )
                            st.image(artifact_path, caption=artifact.path)
                else:
                    st.warning("No MLflow runs found. Train models first.")
            else:
                st.warning(f"Experiment '{experiment_name}' not found in MLflow.")
        except Exception as e:
            st.error(f"Error accessing MLflow: {e}")
            st.info("Make sure MLflow server is running and accessible.")
            
            # Display alternative performance metrics if available
            st.subheader("Current Model Performance")
            
            if st.session_state.predictors:
                # TODO: Add code to display current model metrics
                st.info("Performance metrics from MLflow are not available.")

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