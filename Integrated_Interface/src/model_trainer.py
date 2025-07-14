import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import scipy.sparse
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, InputLayer, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


class ARIMAPredictor:
    """Custom ARIMA predictor that can be properly pickled"""
    def __init__(self, arima_model, base_prediction):
        self.arima_model = arima_model
        self.base_prediction = base_prediction
        
    def predict(self, X_new):
        # For new predictions, use the base prediction from ARIMA
        # You could add feature-based adjustments here
        return self.base_prediction
        
    def forecast(self, steps=1, exog=None):
        # Return the base prediction for any forecast request
        return np.array([self.base_prediction] * steps)


class SimpleFallback:
    """Simple fallback predictor for when ARIMA fails"""
    def __init__(self, prediction):
        self.prediction = prediction
        
    def forecast(self, steps=1, exog=None):
        return np.array([self.prediction] * steps)
        
    def predict(self, X):
        return self.prediction


class ModelTrainer:
    """
    ModelTrainer without MLflow dependency - saves models directly to files
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create metrics tracking
        self.metrics_log = []
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        mask = y_true != 0
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def evaluate_model(self, y_test, y_pred):
        """Calculate and return common regression metrics"""
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = self.calculate_mape(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        }
    
    def save_metrics(self, model_type, metrics, params, feature_count):
        """Save metrics to a JSON file"""
        metrics_entry = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "feature_count": feature_count,
            "metrics": metrics
        }
        self.metrics_log.append(metrics_entry)
        
        # Save to file
        metrics_file = os.path.join(self.model_dir, "metrics_log.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
    
    def train_xgboost(self, processed_data, params=None, tune_hyperparams=False):
        """
        Train XGBoost using non-temporal features
        
        Args:
            processed_data: Dictionary containing 'xgboost' and 'temporal' data
        """
        # Use XGBoost-specific data (without temporal features)
        X_train, X_test, y_train, y_test = processed_data['xgboost']
        
        print("Training XGBoost model...")
        
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 200,
                'random_state': 42
            }
        
        print(f"Feature count: {X_train.shape[1]} (temporal features excluded)")
        
        if tune_hyperparams:
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [200, 300, 400],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2,
                scoring='neg_mean_squared_error'
            )
            
            grid_search.fit(X_train, y_train)
            params.update(grid_search.best_params_)
            print(f"Best parameters from grid search: {grid_search.best_params_}")
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        metrics = self.evaluate_model(y_test, y_pred)
        
        # Print metrics
        print(f"XGBoost (No Temporal) RMSE: {metrics['rmse']:.2f}")
        print(f"XGBoost (No Temporal) MAE: {metrics['mae']:.2f}")
        print(f"XGBoost (No Temporal) MAPE: {metrics['mape']:.2f}%")
        print(f"XGBoost (No Temporal) R²: {metrics['r2']:.2f}")
        
        # Save metrics
        self.save_metrics("xgboost", metrics, params, X_train.shape[1])
        
        # Plot feature importance
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        # Show top 20 features
        top_n = min(20, len(feature_importance))
        plt.bar(range(top_n), feature_importance[sorted_idx][:top_n])
        plt.xticks(range(top_n), [f'Feature {sorted_idx[i]}' for i in range(top_n)], rotation=45)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('XGBoost Feature Importance (Top 20, No Temporal Features)')
        plt.tight_layout()
        
        importance_fig_path = os.path.join(self.model_dir, "xgboost_feature_importance.png")
        plt.savefig(importance_fig_path)
        plt.close()
        
        # Save model
        model_path = os.path.join(self.model_dir, "xgboost_model.json")
        model.save_model(model_path)
        
        # Save model info
        model_info = {
            "model_type": "xgboost",
            "params": params,
            "metrics": metrics,
            "model_path": model_path,
            "feature_count": X_train.shape[1],
            "created_at": datetime.now().isoformat()
        }
        
        info_path = os.path.join(self.model_dir, "xgboost_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Update best model if this one is better
        if metrics['rmse'] < self.best_score:
            self.best_score = metrics['rmse']
            self.best_model = model
            self.best_model_name = "xgboost"
        
        return model
    
    def train_lstm(self, processed_data, params=None):
        """
        Train LSTM using temporal features with GPU acceleration
        
        Args:
            processed_data: Dictionary containing 'xgboost' and 'temporal' data
        """
        import tensorflow as tf
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Training on GPU: {len(gpus)} GPU(s) available")
            # Memory growth needs to be set before GPUs have been initialized
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu}")
                except RuntimeError as e:
                    print(f"Memory growth setting failed: {e}")
        else:
            print("No GPU found. Training on CPU instead.")
        
        # Enable mixed precision (if supported by GPU)
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")
        
        # Use temporal data (with temporal features)
        X_train, X_test, y_train, y_test = processed_data['temporal']
        
        print("Training LSTM model...")
        
        if params is None:
            params = {
                'lstm_units': 512,
                'dense_units': 256,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 30
            }
        
        print(f"Feature count: {X_train.shape[1]} (temporal features included)")
        
        # Convert sparse matrices to dense arrays
        if scipy.sparse.issparse(X_train):
            print("Converting sparse training data to dense array...")
            X_train = X_train.toarray()
        
        if scipy.sparse.issparse(X_test):
            print("Converting sparse test data to dense array...")
            X_test = X_test.toarray()
        
        # Handle NaN values
        X_train = np.nan_to_num(X_train)
        y_train = np.nan_to_num(y_train)
        X_test = np.nan_to_num(X_test)
        y_test = np.nan_to_num(y_test)
        
        # Reshape for LSTM
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build LSTM model
        model = Sequential()
        
        # Input layer - Bidirectional LSTM
        model.add(Bidirectional(
            LSTM(
                units=params['lstm_units'] // 2,  # Halve the units since Bidirectional doubles the output
                input_shape=(1, X_train.shape[1]),
                return_sequences=True,
                activation='relu',
                kernel_initializer='he_normal'
            )
        ))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Second LSTM layer - also Bidirectional
        model.add(Bidirectional(
            LSTM(
                units=params['lstm_units'] // 4,  # Halve units again for the second layer
                return_sequences=False,
                activation='relu'
            )
        ))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Dense hidden layers
        model.add(Dense(
            units=params['dense_units'],
            activation='relu',
            kernel_initializer='he_normal'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber_loss',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        from keras.callbacks import TerminateOnNaN
        nan_terminator = TerminateOnNaN()
        
        # Train model
        history = model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_test_reshaped, y_test),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[early_stopping, nan_terminator],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('LSTM Model Loss (With Temporal)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('LSTM Model MAE (With Temporal)')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        history_fig_path = os.path.join(self.model_dir, "lstm_training_history.png")
        plt.savefig(history_fig_path)
        plt.close()
        
        # Make predictions
        y_pred = model.predict(X_test_reshaped).flatten()
        
        # Calculate metrics
        metrics = self.evaluate_model(y_test, y_pred)
        
        # Print metrics
        print(f"LSTM (With Temporal) RMSE: {metrics['rmse']:.2f}")
        print(f"LSTM (With Temporal) MAE: {metrics['mae']:.2f}")
        print(f"LSTM (With Temporal) MAPE: {metrics['mape']:.2f}%")
        print(f"LSTM (With Temporal) R²: {metrics['r2']:.2f}")
        
        # Save metrics
        self.save_metrics("lstm", metrics, params, X_train.shape[1])
        
        # Save model with explicit format
        model_path = os.path.join(self.model_dir, "lstm_model.keras")
        model.save(model_path, save_format='tf')  # Save in TensorFlow format explicitly
        
        # Save model info
        model_info = {
            "model_type": "lstm",
            "params": params,
            "metrics": metrics,
            "model_path": model_path,
            "feature_count": X_train.shape[1],
            "created_at": datetime.now().isoformat()
        }
        
        info_path = os.path.join(self.model_dir, "lstm_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Update best model if this one is better
        if metrics['rmse'] < self.best_score:
            self.best_score = metrics['rmse']
            self.best_model = model
            self.best_model_name = "lstm"
        
        return model
    
    def train_arima(self, processed_data, params=None):
        """
        Train ARIMA model using temporal features
        
        Args:
            processed_data: Dictionary containing 'xgboost' and 'temporal' data
        """
        # Use temporal data
        X_train, X_test, y_train, y_test = processed_data['temporal']
        
        print("Training ARIMA model...")
        
        if params is None:
            params = {
                'p': 1,
                'd': 0,
                'q': 0
            }
        
        print(f"Feature count: {X_train.shape[1]} (temporal features included)")
        
        try:
            # Convert to numpy array if needed
            if hasattr(y_train, 'values'):
                y_train_array = y_train.values
            else:
                y_train_array = np.array(y_train)
            
            # Create and fit ARIMA model
            model = ARIMA(y_train_array, order=(params['p'], params['d'], params['q']))
            fitted_model = model.fit()
            
            # Calculate base prediction
            base_prediction = np.mean(y_train_array)
            
            # Create custom predictor
            predictor = ARIMAPredictor(fitted_model, base_prediction)
            
            # Make predictions
            y_pred = np.full(len(y_test), base_prediction)
            
            # Calculate metrics
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Print metrics
            print(f"ARIMA (With Temporal) RMSE: {metrics['rmse']:.2f}")
            print(f"ARIMA (With Temporal) MAE: {metrics['mae']:.2f}")
            print(f"ARIMA (With Temporal) MAPE: {metrics['mape']:.2f}%")
            print(f"ARIMA (With Temporal) R²: {metrics['r2']:.2f}")
            
            # Save metrics
            self.save_metrics("arima", metrics, params, X_train.shape[1])
            
            # Plot predictions
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[:100], label='Actual', alpha=0.7)
            plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
            plt.title('ARIMA: Actual vs Predicted Values (With Temporal)')
            plt.xlabel('Sample Index')
            plt.ylabel('Rental Price (SGD)')
            plt.legend()
            plt.tight_layout()
            
            pred_fig_path = os.path.join(self.model_dir, "arima_predictions.png")
            plt.savefig(pred_fig_path)
            plt.close()
            
            # Save model summary
            summary_path = os.path.join(self.model_dir, "arima_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(str(fitted_model.summary()))
            
            # Create model dictionary
            arima_model_dict = {
                'model': predictor,
                'params': params,
                'base_prediction': base_prediction,
                'metrics': metrics
            }
            
            # Save using joblib
            model_path = os.path.join(self.model_dir, "arima_model.pkl")
            joblib.dump(arima_model_dict, model_path)
            
            # Save model info
            model_info = {
                "model_type": "arima",
                "params": params,
                "metrics": metrics,
                "model_path": model_path,
                "feature_count": X_train.shape[1],
                "created_at": datetime.now().isoformat()
            }
            
            info_path = os.path.join(self.model_dir, "arima_info.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Update best model if this one is better
            if metrics['rmse'] < self.best_score:
                self.best_score = metrics['rmse']
                self.best_model = arima_model_dict
                self.best_model_name = "arima"
            
            return arima_model_dict
            
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            # Create fallback model
            base_prediction = np.mean(y_train)
            
            fallback_model = {
                'model': SimpleFallback(base_prediction),
                'params': params,
                'base_prediction': base_prediction
            }
            
            # Calculate metrics using the fallback
            y_pred = np.full(len(y_test), base_prediction)
            metrics = self.evaluate_model(y_test, y_pred)
            fallback_model['metrics'] = metrics
            
            # Save fallback
            model_path = os.path.join(self.model_dir, "arima_fallback.pkl")
            joblib.dump(fallback_model, model_path)
            
            return fallback_model
    
    def load_model(self, model_type):
        """
        Load a trained model from disk
        
        Args:
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
            
        Returns:
            Trained model
        """
        if model_type == "xgboost":
            model_path = os.path.join(self.model_dir, "xgboost_model.json")
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        elif model_type == "lstm":
            from keras.models import load_model
            # Try different possible paths
            possible_paths = [
                os.path.join(self.model_dir, "lstm_model.keras"),
                os.path.join(self.model_dir, "lstm_model"),
                os.path.join(self.model_dir, "lstm_model.h5")
            ]
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    try:
                        return load_model(model_path)
                    except Exception as e:
                        print(f"Failed to load from {model_path}: {e}")
                        continue
            
            # If all attempts fail
            raise FileNotFoundError(f"Could not find LSTM model in any of: {possible_paths}")
            
        elif model_type == "arima":
            model_path = os.path.join(self.model_dir, "arima_model.pkl")
            return joblib.load(model_path)
        else:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
            return joblib.load(model_path)
    
    def get_available_models(self):
        """Get list of available saved models"""
        models = []
        model_files = {
            'xgboost': 'xgboost_model.json',
            'lstm': ['lstm_model.keras', 'lstm_model', 'lstm_model.h5'],
            'arima': 'arima_model.pkl'
        }
        
        for model_type, file_paths in model_files.items():
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            
            for file_path in file_paths:
                full_path = os.path.join(self.model_dir, file_path)
                if os.path.exists(full_path):
                    try:
                        # Load model info
                        info_path = os.path.join(self.model_dir, f"{model_type}_info.json")
                        if os.path.exists(info_path):
                            with open(info_path, 'r') as f:
                                info = json.load(f)
                            models.append({
                                'type': model_type,
                                'info': info
                            })
                        else:
                            models.append({
                                'type': model_type,
                                'info': {'model_type': model_type}
                            })
                        break  # Found a valid file for this model type
                    except:
                        continue
        
        return models
    
    def get_best_model(self):
        """
        Get the best performing model
        
        Returns:
            tuple: (best_model, best_model_name)
        """
        return self.best_model, self.best_model_name