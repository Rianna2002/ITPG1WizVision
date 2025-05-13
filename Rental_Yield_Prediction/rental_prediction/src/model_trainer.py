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

class ModelTrainer:
    """
    Class for training, evaluating, and saving ML models for rental prediction.
    Supports XGBoost, LSTM, ARIMA, and possibility to add other models.
    Includes MLflow tracking integration.
    """
    
    def __init__(self, experiment_name="rental_prediction"):
        """
        Initialize ModelTrainer with MLflow experiment tracking
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        # Set up MLflow
        self.experiment_name = experiment_name
        # Try to get the experiment ID, or create a new experiment if it doesn't exist
        try:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            
        self.best_model = None
        self.best_model_name = None
        self.best_score = float('inf')  # Lower RMSE is better
    
    def calculate_mape(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            float: MAPE value in percentage
        """
        # Avoid division by zero
        mask = y_true != 0
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def evaluate_model(self, y_test, y_pred):
        """
        Calculate and return common regression metrics
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary containing MSE, RMSE, MAE, MAPE, and R²
        """
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
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, params=None, tune_hyperparams=False):
        """
        Train an XGBoost model for rental prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            params (dict): XGBoost parameters
            tune_hyperparams (bool): Whether to perform grid search for hyperparameter tuning
            
        Returns:
            trained XGBoost model
        """
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="XGBoost"):
            
            if params is None:
                params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 200
                }
            
            if tune_hyperparams:
                # Define parameter grid for grid search
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
                
                # Create base model
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
                
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=xgb_model,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    verbose=2,
                    scoring='neg_mean_squared_error'
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best parameters and update params
                params.update(grid_search.best_params_)
                print(f"Best parameters from grid search: {grid_search.best_params_}")
            
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)
                
            # Train model with best parameters
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Make predictions and evaluate
            y_pred = model.predict(X_test)
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"XGBoost RMSE: {metrics['rmse']:.2f}")
            print(f"XGBoost MAE: {metrics['mae']:.2f}")
            print(f"XGBoost MAPE: {metrics['mape']:.2f}%")
            print(f"XGBoost R²: {metrics['r2']:.2f}")
            
            # Plot feature importance
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(X_train.shape[1]), feature_importance[sorted_idx])
            plt.xticks(range(X_train.shape[1]), sorted_idx, rotation=90)
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title('XGBoost Feature Importance')
            plt.tight_layout()
            
            # Save figure
            importance_fig_path = "feature_importance.png"
            plt.savefig(importance_fig_path)
            mlflow.log_artifact(importance_fig_path)
            
            # Log model
            mlflow.xgboost.log_model(model, "xgboost_model")
            
            # Update best model if this one is better
            if metrics['rmse'] < self.best_score:
                self.best_score = metrics['rmse']
                self.best_model = model
                self.best_model_name = "xgboost"
            
            return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, params=None):
        """
        Train an LSTM model for rental prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            params (dict): LSTM parameters
            
        Returns:
            trained LSTM model
        """
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
            
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)
            
            # Convert sparse matrices to dense arrays
            if scipy.sparse.issparse(X_train):
                print("Converting sparse training data to dense array...")
                X_train = X_train.toarray()
            
            if scipy.sparse.issparse(X_test):
                print("Converting sparse test data to dense array...")
                X_test = X_test.toarray()
            
            # Check for NaN or infinite values
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)
            
            # Reshape input for LSTM [samples, timesteps, features]
            # For simplicity, we'll use timesteps=1 since this is not a time series model in the traditional sense
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
                clipnorm=1.0,  # Clip gradients to prevent explosion
                clipvalue=0.5  # Additional safeguard
            )
            
            model.compile(
                optimizer=optimizer,
                loss='huber_loss',  # Robust to outliers
                metrics=['mae']
            )
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            )
            
            # Add NaN terminator
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
            plt.title('LSTM Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('LSTM Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            # Save and log training history plot
            history_fig_path = "lstm_training_history.png"
            plt.savefig(history_fig_path)
            mlflow.log_artifact(history_fig_path)
            
            # Make predictions
            y_pred = model.predict(X_test_reshaped).flatten()
            
            # Calculate metrics
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"LSTM RMSE: {metrics['rmse']:.2f}")
            print(f"LSTM MAE: {metrics['mae']:.2f}")
            print(f"LSTM MAPE: {metrics['mape']:.2f}%")
            print(f"LSTM R²: {metrics['r2']:.2f}")
            
            # Log model
            mlflow.keras.log_model(model, "lstm_model")
            
            # Update best model if this one is better
            if metrics['rmse'] < self.best_score:
                self.best_score = metrics['rmse']
                self.best_model = model
                self.best_model_name = "lstm"
            
            return model
    
    def train_arima(self, X_train, y_train, X_test, y_test, params=None):
        """
        Train an ARIMA model for rental prediction
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            params (dict): ARIMA parameters
            
        Returns:
            trained ARIMA model
        """
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="ARIMA"):
            
            if params is None:
                params = {
                    'p': 2,  # AR order
                    'd': 1,  # Differencing order
                    'q': 2   # MA order
                }
            
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)
            
            # Note: Traditional ARIMA is for time series and doesn't directly use feature data.
            # This is a simplified approach that builds individual ARIMA models for each test sample
            # using exogenous variables (X). It's more of a demonstration than a recommended approach.
            
            # Function to create and fit an ARIMA model with exogenous variables
            def fit_arima_with_exog(y, X, params):
                """
                Create and fit an ARIMA model with exogenous variables
                
                Args:
                    y: Target variable
                    X: Exogenous variables
                    params: ARIMA parameters (p, d, q)
                    
                Returns:
                    Fitted ARIMA model
                """
                import numpy as np
                import scipy.sparse
                from statsmodels.tsa.arima.model import ARIMA
                
                # Ensure X is a 2D array with proper shape
                if isinstance(X, tuple):
                    # If X is a tuple, convert to numpy array
                    X = np.array(X).reshape(-1, 1)
                elif isinstance(X, np.ndarray):
                    if X.ndim == 1:
                        # If X is a 1D array, reshape it to a 2D array
                        X = X.reshape(-1, 1)
                elif scipy.sparse.issparse(X):
                    # If X is a sparse matrix, convert it to a dense array
                    X = X.toarray()
                    
                # Additional check to ensure it's 2D
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                    
                # Ensure y is a 1D array
                if isinstance(y, np.ndarray) and y.ndim > 1:
                    y = y.flatten()
                
                try:
                    # Create and fit the ARIMA model
                    model = ARIMA(
                        y,
                        order=(params['p'], params['d'], params['q']),
                        exog=X
                    )
                    
                    fit = model.fit()
                    return fit
                except Exception as e:
                    print(f"Error fitting ARIMA model: {e}")
                    # Fallback to simpler model
                    try:
                        fallback_model = ARIMA(
                            y,
                            order=(1, 0, 0),  # Simple AR(1) model
                            exog=X
                        )
                        return fallback_model.fit()
                    except Exception as e2:
                        print(f"Error fitting fallback ARIMA model: {e2}")
                        # Last resort: try without exogenous variables
                        try:
                            simple_model = ARIMA(
                                y,
                                order=(1, 0, 0)  # Simple AR(1) model without exog
                            )
                            return simple_model.fit()
                        except Exception as e3:
                            print(f"Could not fit any ARIMA model: {e3}")
                            raise
            
            # ARIMA models typically work with time series data
            # Since our data may not be a time series, we'll use a simplified approach
            
            # Train/fit an ARIMA model on the entire training data
            # Note: This is a simplified approach and may not be optimal
            model = fit_arima_with_exog(y_train, X_train, params)
            
            # Store the ARIMA model and parameters for later use
            arima_model = {
                'model': model,
                'params': params,
                'features': X_train.shape[1]  # Store feature count for consistency checks
            }
            
            # Make predictions on test data
            try:
                y_pred = model.forecast(steps=len(y_test), exog=X_test)
            except Exception as e:
                print(f"Error forecasting with ARIMA: {e}")
                # Fallback prediction (mean of training data)
                y_pred = np.full(len(y_test), np.mean(y_train))
            
            # Calculate metrics
            metrics = self.evaluate_model(y_test, y_pred)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            print(f"ARIMA RMSE: {metrics['rmse']:.2f}")
            print(f"ARIMA MAE: {metrics['mae']:.2f}")
            print(f"ARIMA MAPE: {metrics['mape']:.2f}%")
            print(f"ARIMA R²: {metrics['r2']:.2f}")
            
            # Plot actual vs predicted values
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[:100], label='Actual')  # Limit to first 100 for visibility
            plt.plot(y_pred[:100], label='Predicted')
            plt.title('ARIMA: Actual vs Predicted Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            
            # Save and log plot
            pred_fig_path = "arima_predictions.png"
            plt.savefig(pred_fig_path)
            mlflow.log_artifact(pred_fig_path)
            
            # Save the model parameters and summary
            summary_path = "arima_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(str(model.summary()))
            mlflow.log_artifact(summary_path)
            
            # Log model as a custom Python object
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="arima_model"
            )
            
            # Update best model if this one is better
            if metrics['rmse'] < self.best_score:
                self.best_score = metrics['rmse']
                self.best_model = arima_model
                self.best_model_name = "arima"
            
            return arima_model
    
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