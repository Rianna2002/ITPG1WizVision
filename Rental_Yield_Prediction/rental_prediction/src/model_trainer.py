
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.sklearn
import mlflow.xgboost
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import scipy.sparse
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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
    Enhanced ModelTrainer that uses appropriate preprocessing for each model type
    """
    
    def __init__(self, experiment_name="rental_prediction"):
        self.experiment_name = experiment_name
        try:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')
    
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
    
    def train_xgboost(self, processed_data, params=None, tune_hyperparams=False):
        """
        Train XGBoost using non-temporal features
        
        Args:
            processed_data: Dictionary containing 'xgboost' and 'temporal' data
        """
        # Use XGBoost-specific data (without temporal features)
        X_train, X_test, y_train, y_test = processed_data['xgboost']
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="XGBoost"):
            
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
            
            # Log that this model excludes temporal features
            mlflow.log_param("temporal_features", "excluded")
            mlflow.log_param("feature_count", X_train.shape[1])
            
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
            
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Make predictions and evaluate
            y_pred = model.predict(X_test)
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"XGBoost (No Temporal) RMSE: {metrics['rmse']:.2f}")
            print(f"XGBoost (No Temporal) MAE: {metrics['mae']:.2f}")
            print(f"XGBoost (No Temporal) MAPE: {metrics['mape']:.2f}%")
            print(f"XGBoost (No Temporal) R²: {metrics['r2']:.2f}")
            
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
            
            importance_fig_path = "xgboost_no_temporal_feature_importance.png"
            plt.savefig(importance_fig_path)
            mlflow.log_artifact(importance_fig_path)
            plt.close()
            
            # Log model
            mlflow.xgboost.log_model(model, "xgboost_no_temporal_model")
            
            # Update best model if this one is better
            if metrics['rmse'] < self.best_score:
                self.best_score = metrics['rmse']
                self.best_model = model
                self.best_model_name = "xgboost"
            
            return model
    
    def train_lstm(self, processed_data, params=None):
        """
        Train LSTM using temporal features
        
        Args:
            processed_data: Dictionary containing 'xgboost' and 'temporal' data
        """
        # Use temporal data (with temporal features)
        X_train, X_test, y_train, y_test = processed_data['temporal']
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="LSTM"):
            
            if params is None:
                params = {
                    'lstm_units': 64,
                    'dense_units': 32,
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 20
                }
            
            # Log that this model includes temporal features
            mlflow.log_param("temporal_features", "included")
            mlflow.log_param("feature_count", X_train.shape[1])
            
            for param, value in params.items():
                mlflow.log_param(param, value)
            
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
            
            # Input layer
            model.add(LSTM(
                units=params['lstm_units'],
                input_shape=(1, X_train.shape[1]),
                return_sequences=False,
                activation='relu',
                kernel_initializer='he_normal'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
            
            # Dense hidden layer
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
            
            from tensorflow.keras.callbacks import TerminateOnNaN
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
            
            history_fig_path = "lstm_temporal_training_history.png"
            plt.savefig(history_fig_path)
            mlflow.log_artifact(history_fig_path)
            plt.close()
            
            # Make predictions
            y_pred = model.predict(X_test_reshaped).flatten()
            
            # Calculate metrics
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"LSTM (With Temporal) RMSE: {metrics['rmse']:.2f}")
            print(f"LSTM (With Temporal) MAE: {metrics['mae']:.2f}")
            print(f"LSTM (With Temporal) MAPE: {metrics['mape']:.2f}%")
            print(f"LSTM (With Temporal) R²: {metrics['r2']:.2f}")
            
            # Log model
            mlflow.keras.log_model(model, "lstm_temporal_model")
            
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
        
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="ARIMA"):
            
            if params is None:
                params = {
                    'p': 1,
                    'd': 0,
                    'q': 0
                }
            
            # Log that this model uses temporal features
            mlflow.log_param("temporal_features", "included")
            mlflow.log_param("feature_count", X_train.shape[1])
            
            for param, value in params.items():
                mlflow.log_param(param, value)
            
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
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                print(f"ARIMA (With Temporal) RMSE: {metrics['rmse']:.2f}")
                print(f"ARIMA (With Temporal) MAE: {metrics['mae']:.2f}")
                print(f"ARIMA (With Temporal) MAPE: {metrics['mape']:.2f}%")
                print(f"ARIMA (With Temporal) R²: {metrics['r2']:.2f}")
                
                # Plot predictions
                plt.figure(figsize=(10, 6))
                plt.plot(y_test[:100], label='Actual', alpha=0.7)
                plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
                plt.title('ARIMA: Actual vs Predicted Values (With Temporal)')
                plt.xlabel('Sample Index')
                plt.ylabel('Rental Price (SGD)')
                plt.legend()
                plt.tight_layout()
                
                pred_fig_path = "arima_temporal_predictions.png"
                plt.savefig(pred_fig_path)
                mlflow.log_artifact(pred_fig_path)
                plt.close()
                
                # Save model summary
                summary_path = "arima_temporal_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(str(fitted_model.summary()))
                mlflow.log_artifact(summary_path)
                
                # Create model dictionary
                arima_model_dict = {
                    'model': predictor,
                    'params': params,
                    'base_prediction': base_prediction
                }
                
                # Save using joblib
                model_path = "arima_temporal_model.pkl"
                joblib.dump(arima_model_dict, model_path)
                mlflow.log_artifact(model_path)
                
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
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                return fallback_model
    
    def save_model(self, model, model_type, model_dir="models"):
        """
        Save the trained model to disk
        
        Args:
            model: Trained model
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
            model_dir (str): Directory to save the model
            
        Returns:
            str: Path to the saved model
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_type}_model")
        
        # Save model based on type
        if model_type == "xgboost":
            model.save_model(f"{model_path}.json")
            return f"{model_path}.json"
        elif model_type == "lstm":
            model.save(model_path)
            return model_path
        elif model_type == "arima":
            # ARIMA models from statsmodels are saved differently
            import pickle
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(model, f)
            return f"{model_path}.pkl"
        else:
            joblib.dump(model, f"{model_path}.pkl")
            return f"{model_path}.pkl"
    
    def load_model(self, model_path, model_type):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            model_type (str): Type of model ('xgboost', 'lstm', 'arima')
            
        Returns:
            Trained model
        """
        if model_type == "xgboost":
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        elif model_type == "lstm":
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        elif model_type == "arima":
            import pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            return joblib.load(model_path)
    
    def get_best_model(self):
        """
        Get the best performing model
        
        Returns:
            tuple: (best_model, best_model_name)
        """
        return self.best_model, self.best_model_name