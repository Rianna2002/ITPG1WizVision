import pandas as pd
import numpy as np
import os
import json
import scipy.sparse
import joblib
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer


class RentalPredictor:
    """
    Enhanced RentalPredictor that handles different feature sets for different models
    without MLflow dependency
    """
    
    def __init__(self, preprocessor=None, model=None, model_type=None):
        """
        Initialize the RentalPredictor
        
        Args:
            preprocessor: DataProcessor instance with both preprocessors
            model: Trained model
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
        """
        self.preprocessor = preprocessor
        self.model = model
        self.model_type = model_type
        
    def load_from_file(self, model_type, model_dir="models"):
        """Load a model from file with proper handling for each model type"""
        try:
            if model_type == "xgboost":
                import xgboost as xgb
                model_path = os.path.join(model_dir, "xgboost_model.json")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"XGBoost model not found at {model_path}")
                self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
                
            elif model_type == "lstm":
                from tensorflow.keras.models import load_model
                # Try different possible paths for LSTM model
                possible_paths = [
                    os.path.join(model_dir, "lstm_model.keras"),
                    os.path.join(model_dir, "lstm_model"),
                    os.path.join(model_dir, "lstm_model.h5")
                ]
                
                model_loaded = False
                for model_path in possible_paths:
                    if os.path.exists(model_path):
                        try:
                            print(f"Attempting to load LSTM from: {model_path}")
                            self.model = load_model(model_path)
                            print(f"Successfully loaded LSTM model")
                            print(f"Model type: {type(self.model)}")
                            print(f"Model has predict method: {hasattr(self.model, 'predict')}")
                            model_loaded = True
                            break
                        except Exception as e:
                            print(f"Failed to load from {model_path}: {e}")
                            continue
                
                if not model_loaded:
                    raise FileNotFoundError(f"Could not load LSTM model from any of: {possible_paths}")
                
            elif model_type == "arima":
                model_path = os.path.join(model_dir, "arima_model.pkl")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"ARIMA model not found at {model_path}")
                self.model = joblib.load(model_path)
                
            else:
                model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found at {model_path}")
                self.model = joblib.load(model_path)
            
            self.model_type = model_type
            
            # Load model info if available
            info_path = os.path.join(model_dir, f"{model_type}_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            
            print(f"Model loaded successfully: {model_type}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, input_data):
        """
        Make a rental prediction for the given input data
        
        Args:
            input_data (dict): Dictionary containing user input
        """
        if self.preprocessor is None or self.model is None:
            raise ValueError("Preprocessor and model must be set before making predictions")
        
        print(f"[DEBUG] Making prediction with {self.model_type}")
        print(f"[DEBUG] Input data: {input_data}")
        
        # Determine which preprocessor to use based on model type
        preprocess_model_type = 'xgboost' if self.model_type == 'xgboost' else 'temporal'
        
        # Preprocess input data
        processed_input = self.preprocessor.preprocess_input(input_data, preprocess_model_type)
        
        print(f"[DEBUG] Processed input shape: {processed_input.shape}")
        
        # Convert sparse matrix to dense array if needed
        if scipy.sparse.issparse(processed_input):
            processed_input = processed_input.toarray()
        
        # Ensure processed_input is 2D
        if processed_input.ndim == 1:
            processed_input = processed_input.reshape(1, -1)
        
        # Make prediction based on model type
        if self.model_type == "xgboost":
            print(f"[DEBUG] Using XGBoost (no temporal features)")
            prediction_array = self.model.predict(processed_input)
            prediction = prediction_array[0]
            
            # Test different inputs to verify model is working
            print(f"[DEBUG] XGBoost prediction: {prediction}")
            
            # Test with zeros to see different prediction
            test_zeros = np.zeros_like(processed_input)
            test_pred_zeros = self.model.predict(test_zeros)
            print(f"[DEBUG] XGBoost with zeros: {test_pred_zeros[0]}")
            
        elif self.model_type == "lstm":
            print(f"[DEBUG] Using LSTM (with temporal features)")
            # Ensure model is loaded correctly
            if not hasattr(self.model, 'predict'):
                raise ValueError(f"LSTM model object does not have 'predict' method. Model type: {type(self.model)}")
                
            # Reshape for LSTM [samples, timesteps, features]
            lstm_input = processed_input.reshape((processed_input.shape[0], 1, processed_input.shape[1]))
            print(f"[DEBUG] LSTM input shape: {lstm_input.shape}")
            
            prediction = self.model.predict(lstm_input, verbose=0)[0][0]
            print(f"[DEBUG] LSTM prediction: {prediction}")
            
        elif self.model_type == "arima":
            print(f"[DEBUG] Using ARIMA (with temporal features)")
            try:
                # Handle different ARIMA model formats
                if isinstance(self.model, dict) and 'model' in self.model:
                    arima_model = self.model['model']
                    # Use the forecast method
                    prediction = arima_model.forecast(steps=1)[0]
                elif hasattr(self.model, 'forecast'):
                    prediction = self.model.forecast(steps=1)[0]
                else:
                    # Fallback to simple baseline
                    prediction = 2500.0
                    print("[WARNING] Using fallback prediction for ARIMA")
                    
                # Ensure valid prediction
                if prediction <= 0 or np.isnan(prediction):
                    prediction = 2500.0
                    print("[WARNING] ARIMA prediction was invalid, using fallback")
                    
            except Exception as e:
                print(f"[ERROR] ARIMA prediction failed: {e}")
                prediction = 2500.0
                
        else:
            # Generic model
            prediction = self.model.predict(processed_input)[0]
        
        # Ensure prediction is a positive scalar
        if isinstance(prediction, (list, np.ndarray)):
            prediction = float(prediction[0]) if len(prediction) > 0 else 2500.0
        else:
            prediction = float(prediction)
        
        # Ensure prediction is positive
        if prediction <= 0:
            prediction = 2500.0
            
        print(f"[DEBUG] Final prediction for {self.model_type}: {prediction}")
        return prediction
    
    def save_model(self, model_dir="models"):
        """Save the model and preprocessor"""
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model (the model should already be saved by ModelTrainer,
        # but we'll ensure it's saved here as well)
        if self.model is not None:
            if self.model_type == "xgboost":
                model_path = os.path.join(model_dir, "xgboost_model.json")
                self.model.save_model(model_path)
                
            elif self.model_type == "lstm":
                model_path = os.path.join(model_dir, "lstm_model.keras")
                self.model.save(model_path, save_format='tf')
                
            elif self.model_type == "arima":
                model_path = os.path.join(model_dir, "arima_model.pkl")
                joblib.dump(self.model, model_path)
                
            else:
                model_path = os.path.join(model_dir, f"{self.model_type}_model.pkl")
                joblib.dump(self.model, model_path)
                
            print(f"Model saved: {model_path}")
        
        # Save preprocessors
        if self.preprocessor is not None:
            # Save both specific preprocessors
            xgb_preprocessor_path = os.path.join(model_dir, "preprocessor_xgboost.pkl")
            temporal_preprocessor_path = os.path.join(model_dir, "preprocessor_temporal.pkl")
            
            if hasattr(self.preprocessor, 'preprocessor_xgboost'):
                joblib.dump(self.preprocessor.preprocessor_xgboost, xgb_preprocessor_path)
                print(f"XGBoost preprocessor saved: {xgb_preprocessor_path}")
            if hasattr(self.preprocessor, 'preprocessor_temporal'):
                joblib.dump(self.preprocessor.preprocessor_temporal, temporal_preprocessor_path)
                print(f"Temporal preprocessor saved: {temporal_preprocessor_path}")
            
            # Save the full data processor
            full_preprocessor_path = os.path.join(model_dir, "data_processor.pkl")
            joblib.dump(self.preprocessor, full_preprocessor_path)
            print(f"Data processor saved: {full_preprocessor_path}")
        
        # Get model path based on type (for return value)
        if self.model_type == "xgboost":
            model_path = os.path.join(model_dir, "xgboost_model.json")
        elif self.model_type == "lstm":
            model_path = os.path.join(model_dir, "lstm_model.keras")
        elif self.model_type == "arima":
            model_path = os.path.join(model_dir, "arima_model.pkl")
        else:
            model_path = os.path.join(model_dir, f"{self.model_type}_model.pkl")
        
        full_preprocessor_path = os.path.join(model_dir, "data_processor.pkl")
        return model_path, full_preprocessor_path
    
    def load_preprocessor(self, preprocessor_path):
        """Load a preprocessor from file"""
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor loaded successfully from {preprocessor_path}")
            return True
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return False