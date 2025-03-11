import streamlit as st
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.xgboost
import mlflow.keras
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

# Initialize session state variables if they don't exist
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
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
def train_model(data_processor, X_train, X_test, y_train, y_test, model_type, tune_hyperparams=False):
    trainer = ModelTrainer()
    
    if model_type == "xgboost":
        model = trainer.train_xgboost(X_train, y_train, X_test, y_test, tune_hyperparams=tune_hyperparams)
    elif model_type == "mlp":
        model = trainer.train_mlp(X_train, y_train, X_test, y_test)
    else:
        st.error("Unsupported model type")
        return None
    
    best_model, best_model_name = trainer.get_best_model()
    
    # Create predictor
    predictor = RentalPredictor(
        preprocessor=data_processor,
        model=best_model,
        model_type=best_model_name
    )
    
    # Save model
    model_path, preprocessor_path = predictor.save_model()
    
    return predictor

# Function to make prediction
def predict_rent(predictor, input_features):
    prediction = predictor.predict(input_features)
    return prediction

# Prediction Page
if page == "Predict":
    st.header("Predict Monthly Rent")
    
    # Check if model is loaded
    if not st.session_state.model_loaded and not st.session_state.model_trained:
        st.warning("No model is loaded. Please go to 'Train Model' page to train a new model or load an existing one.")
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
                
                # Make prediction
                predicted_rent = predict_rent(st.session_state.predictor, input_features)
                
                # Display result
                st.success(f"Predicted Monthly Rent: S${predicted_rent:.2f}")
                
                # Additional analysis
                st.subheader("Rental Analysis")
                
                # Calculate annual rent
                annual_rent = predicted_rent * 12
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Monthly Rent", f"S${predicted_rent:.2f}")
                    st.metric("Annual Rent", f"S${annual_rent:.2f}")
                
                with col2:
                    # Compare with average rent for that town and flat type
                    if os.path.exists(data_path):
                        avg_rent = data[(data['town'] == town) & (data['flat_type'] == flat_type)]['monthly_rent'].mean()
                        diff = predicted_rent - avg_rent
                        st.metric(f"Average Rent for {flat_type} in {town}", f"S${avg_rent:.2f}", f"{diff:+.2f}")
                        
                        # Comparison with similar properties
                        st.markdown("### Similar Properties")
                        similar = data[(data['town'] == town) & (data['flat_type'] == flat_type)].head(5)
                        st.dataframe(similar[['block', 'street_name', 'monthly_rent']])

# Train Model Page
elif page == "Train Model":
    st.header("Train or Load a Model")
    
    tab1, tab2 = st.tabs(["Train New Model", "Load Existing Model"])
    
    with tab1:
        st.subheader("Train a New Rental Prediction Model")
        
        # Data upload or selection
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
            model_type = st.selectbox("Select Model Type", ["xgboost", "mlp"])
            
            # Hyperparameter tuning option
            tune_hyperparams = st.checkbox("Tune Hyperparameters", value=False)
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Training model... This may take a while."):
                    # Initialize data processor
                    data_processor = DataProcessor()
                    
                    # Preprocess data
                    X_train, X_test, y_train, y_test = data_processor.preprocess_data(data)
                    
                    # Train model
                    predictor = train_model(
                        data_processor, X_train, X_test, y_train, y_test, 
                        model_type, tune_hyperparams
                    )
                    
                    if predictor:
                        st.session_state.predictor = predictor
                        st.session_state.data_processor = data_processor
                        st.session_state.model_trained = True
                        st.session_state.model_loaded = True
                        
                        st.success("Model trained successfully! Go to the 'Predict' page to make predictions.")
    
    with tab2:
        st.subheader("Load Existing Model")
        
        # MLflow or file-based model loading
        load_option = st.radio("Load model from:", ["MLflow", "Local File"])
        
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
                        
                        selected_run = st.selectbox("Select MLflow Run", run_names)
                        selected_run_id = run_ids[run_names.index(selected_run)]
                        
                        model_type = st.selectbox("Model Type", ["xgboost", "mlp"])
                        
                        if st.button("Load Model"):
                            with st.spinner("Loading model from MLflow..."):
                                predictor = RentalPredictor()
                                success = predictor.load_from_mlflow(selected_run_id, model_type)
                                
                                if success:
                                    # Load preprocessor
                                    data_processor = DataProcessor()
                                    # We would need to load the preprocessor from MLflow artifacts
                                    # For simplicity, we'll initialize a new one and process the data
                                    if os.path.exists(data_path):
                                        data = load_data(data_path)
                                        data_processor.preprocess_data(data)
                                        
                                        predictor.preprocessor = data_processor
                                        st.session_state.predictor = predictor
                                        st.session_state.data_processor = data_processor
                                        st.session_state.model_loaded = True
                                        
                                        st.success("Model loaded successfully! Go to the 'Predict' page to make predictions.")
                                else:
                                    st.error("Failed to load model from MLflow.")
                    else:
                        st.warning("No MLflow runs found. Train a model first.")
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
                    selected_model = st.selectbox("Select Model File", model_files)
                    model_path = os.path.join(model_dir, selected_model)
                    
                    # Determine model type from filename
                    if "xgboost" in selected_model:
                        model_type = "xgboost"
                    elif "mlp" in selected_model:
                        model_type = "mlp"
                    else:
                        model_type = st.selectbox("Model Type", ["xgboost", "mlp"])
                    
                    # Check for preprocessor
                    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
                    
                    if os.path.exists(preprocessor_path):
                        if st.button("Load Model"):
                            with st.spinner("Loading model from file..."):
                                predictor = RentalPredictor()
                                success = predictor.load_from_file(model_path, model_type)
                                
                                if success:
                                    # Load preprocessor
                                    preprocessor_success = predictor.load_preprocessor(preprocessor_path)
                                    
                                    if preprocessor_success:
                                        st.session_state.predictor = predictor
                                        st.session_state.model_loaded = True
                                        
                                        st.success("Model loaded successfully! Go to the 'Predict' page to make predictions.")
                                    else:
                                        st.error("Failed to load preprocessor.")
                                else:
                                    st.error("Failed to load model from file.")
                    else:
                        st.warning("Preprocessor not found. You need both the model and preprocessor to make predictions.")
                else:
                    st.warning("No model files found in the models directory. Train a model first.")
            else:
                st.warning("Models directory not found. Train a model first.")

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    if not st.session_state.model_trained and not st.session_state.model_loaded:
        st.warning("No model has been trained or loaded yet. Please go to the 'Train Model' page first.")
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
                    
                    metrics_df = runs[['tags.mlflow.runName', 'metrics.rmse', 'metrics.mae', 'metrics.r2', 'tags.mlflow.source.name']]
                    metrics_df.columns = ['Model', 'RMSE', 'MAE', 'R²', 'Source']
                    
                    st.dataframe(metrics_df)
                    
                    # Plot metrics comparison
                    st.subheader("Metrics Comparison")
                    
                    # RMSE Comparison
                    st.bar_chart(metrics_df.set_index('Model')['RMSE'])
                    
                    # R² Comparison
                    st.bar_chart(metrics_df.set_index('Model')['R²'])
                    
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
                    st.warning("No MLflow runs found. Train a model first.")
            else:
                st.warning(f"Experiment '{experiment_name}' not found in MLflow.")
        except Exception as e:
            st.error(f"Error accessing MLflow: {e}")
            st.info("Make sure MLflow server is running and accessible.")
            
            # Display alternative performance metrics if available
            st.subheader("Current Model Performance")
            
            if st.session_state.predictor:
                # TODO: Add code to display current model metrics
                st.info("Performance metrics from MLflow are not available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application predicts rental prices for properties in Singapore. "
    "It uses machine learning models trained on historical rental data."
)
st.sidebar.markdown("### Instructions")
st.sidebar.info(
    "1. Train a model or load an existing one from the 'Train Model' page\n"
    "2. Go to the 'Predict' page to make rental predictions\n"
    "3. Check model performance metrics on the 'Model Performance' page"
)