import pandas as pd
import numpy as np
import mlflow
import os
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer

class RentalPredictor:
    """
    Class for making rental predictions using trained models.
    Provides functionality to load trained models and make predictions.
    """
    
    def __init__(self, preprocessor=None, model=None, model_type=None):
        """
        Initialize the RentalPredictor
        
        Args:
            preprocessor: Fitted data preprocessor
            model: Trained model
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
        """
        self.preprocessor = preprocessor
        self.model = model
        self.model_type = model_type
        
    def load_from_mlflow(self, run_id, model_type):
        """
        Load a model from MLflow
        
        Args:
            run_id (str): MLflow run ID
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_type == "xgboost":
                model_path = f"runs:/{run_id}/xgboost_model"
                self.model = mlflow.xgboost.load_model(model_path)
            elif model_type == "lstm":
                model_path = f"runs:/{run_id}/lstm_model"
                self.model = mlflow.keras.load_model(model_path)
            elif model_type == "arima":
                model_path = f"runs:/{run_id}/arima_model"
                self.model = mlflow.sklearn.load_model(model_path)
            else:
                model_path = f"runs:/{run_id}/{model_type}_model"
                self.model = mlflow.sklearn.load_model(model_path)
                
            self.model_type = model_type
            print(f"Model loaded successfully from MLflow run {run_id}")
            return True
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return False
    
    def load_from_file(self, model_path, model_type, preprocessor=None):
        """
        Load a model from file
        
        Args:
            model_path (str): Path to the saved model
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
            preprocessor: Optional data preprocessor
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            trainer = ModelTrainer()
            self.model = trainer.load_model(model_path, model_type)
            self.model_type = model_type
            
            if preprocessor:
                self.preprocessor = preprocessor
                
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model from file: {e}")
            return False
    
    def predict(self, input_data):
        """
        Make a rental prediction for the given input data
        
        Args:
            input_data (dict): Dictionary containing user input (town, flat_type, etc.)
            
        Returns:
            float: Predicted monthly rent
        """
        if self.preprocessor is None or self.model is None:
            raise ValueError("Preprocessor and model must be set before making predictions")
        
        # Preprocess input data
        processed_input = self.preprocessor.preprocess_input(input_data)
        
        # Make prediction based on model type
        if self.model_type == "xgboost":
            prediction = self.model.predict(processed_input)[0]
        elif self.model_type == "lstm":
            # Reshape input for LSTM [samples, timesteps, features]
            lstm_input = processed_input.reshape((1, 1, processed_input.shape[1]))
            prediction = self.model.predict(lstm_input)[0][0]
        elif self.model_type == "arima":
            # For ARIMA models loaded from statsmodels
            try:
                # Check if it's a simple statsmodels ARIMA results object
                if hasattr(self.model, 'forecast'):
                    # Direct forecast
                    prediction = self.model.forecast(steps=1, exog=processed_input)[0]
                # If it's our custom ARIMA dictionary from training
                elif isinstance(self.model, dict) and 'model' in self.model:
                    prediction = self.model['model'].forecast(steps=1, exog=processed_input)[0]
                else:
                    # Fallback - return mean value
                    prediction = processed_input.mean()
            except Exception as e:
                print(f"Error predicting with ARIMA model: {e}")
                # Fallback prediction
                prediction = float(np.mean(processed_input))
        else:
            # Generic model
            prediction = self.model.predict(processed_input)[0]
        
        # Ensure prediction is a scalar
        if isinstance(prediction, (list, np.ndarray)):
            prediction = float(prediction)
        
        return prediction
    
    def save_model(self, model_dir="models"):
        """
        Save the model and preprocessor
        
        Args:
            model_dir (str): Directory to save the model
            
        Returns:
            tuple: (model_path, preprocessor_path)
        """
        import joblib
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        trainer = ModelTrainer()
        model_path = trainer.save_model(self.model, self.model_type, model_dir)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        joblib.dump(self.preprocessor, preprocessor_path)
        
        return model_path, preprocessor_path
    
    def load_preprocessor(self, preprocessor_path):
        """
        Load a preprocessor from file
        
        Args:
            preprocessor_path (str): Path to the saved preprocessor
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import joblib
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor loaded successfully from {preprocessor_path}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return False