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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import joblib
import scipy.sparse

class ModelTrainer:
    """
    Class for training, evaluating, and saving ML models for rental prediction.
    Supports XGBoost, Multi-Layer Perceptron, and possibility to add other models.
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
        self.best_score = float('inf')  # Lower MSE is better
    
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
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            print(f"XGBoost RMSE: {rmse:.2f}")
            print(f"XGBoost MAE: {mae:.2f}")
            print(f"XGBoost R²: {r2:.2f}")
            
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
            if rmse < self.best_score:
                self.best_score = rmse
                self.best_model = model
                self.best_model_name = "xgboost"
            
            return model
    
    def train_mlp(self, X_train, y_train, X_test, y_test, hidden_layers=None):
        """
        Train a Multi-Layer Perceptron (Neural Network) model for rental prediction
        with improved stability measures to prevent NaN loss values.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            hidden_layers (list): List of integers specifying nodes in each hidden layer
            
        Returns:
            trained MLP model
        """
        with mlflow.start_run(experiment_id=self.experiment_id, run_name="MLP"):
            
            # Define model architecture
            if hidden_layers is None:
                hidden_layers = [64, 32, 16]  # Smaller network can be more stable
                
            input_dim = X_train.shape[1]
            
            # Log parameters
            mlflow.log_param("input_dim", input_dim)
            for i, layer_size in enumerate(hidden_layers):
                mlflow.log_param(f"hidden_layer_{i+1}", layer_size)
            
            # Convert sparse matrices to dense arrays
            if scipy.sparse.issparse(X_train):
                print("Converting sparse training data to dense array...")
                X_train = X_train.toarray()
            
            if scipy.sparse.issparse(X_test):
                print("Converting sparse test data to dense array...")
                X_test = X_test.toarray()
            
            # Check for NaN values in data
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                print("Warning: NaN values found in training data. Cleaning...")
                # Replace NaN with 0 or mean values
                X_train = np.nan_to_num(X_train)
                y_train = np.nan_to_num(y_train)
            
            if np.isnan(X_test).any() or np.isnan(y_test).any():
                print("Warning: NaN values found in test data. Cleaning...")
                X_test = np.nan_to_num(X_test)
                y_test = np.nan_to_num(y_test)
            
            # Check for infinite values
            if np.isinf(X_train).any():
                print("Warning: Infinite values found in training data. Replacing with large values...")
                X_train = np.clip(X_train, -1e9, 1e9)
            
            if np.isinf(X_test).any():
                print("Warning: Infinite values found in test data. Replacing with large values...")
                X_test = np.clip(X_test, -1e9, 1e9)
            
            # Create model with a kernel initializer less prone to exploding gradients
            model = Sequential()
            
            # Input layer with careful initialization and batch normalization
            from tensorflow.keras.layers import BatchNormalization
            model.add(Dense(hidden_layers[0], 
                        input_dim=input_dim, 
                        activation='relu',
                        kernel_initializer='he_normal',  # Better for ReLU
                        kernel_regularizer='l2'))  # L2 regularization
            model.add(BatchNormalization())  # Normalize activations
            model.add(Dropout(0.2))
            
            # Hidden layers
            for layer_size in hidden_layers[1:]:
                model.add(Dense(layer_size, 
                            activation='relu',
                            kernel_initializer='he_normal',
                            kernel_regularizer='l2'))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                
            # Output layer - linear activation for regression
            model.add(Dense(1))
            
            # Use a more robust optimizer with gradient clipping
            from tensorflow.keras.optimizers import Adam
            optimizer = Adam(
                learning_rate=0.001,
                clipnorm=1.0,  # Clip gradients to prevent explosion
                clipvalue=0.5  # Additional safeguard against extreme values
            )
            
            # Compile model with mean absolute error as it's more robust than MSE
            model.compile(
                loss='huber_loss',  # Huber loss is less sensitive to outliers than MSE
                optimizer=optimizer, 
                metrics=['mae']
            )
            
            # Define early stopping with more patience
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                verbose=1,
                restore_best_weights=True,
                min_delta=0.001  # Minimum improvement to be considered significant
            )
            
            # Add a callback to stop if NaN loss occurs
            from tensorflow.keras.callbacks import TerminateOnNaN
            nan_terminator = TerminateOnNaN()
            
            # Train model with smaller batch size and careful validation split
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=64,  # Larger batch size for more stable gradients
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, nan_terminator],
                verbose=1
            )
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            # Save history plot
            history_fig_path = "training_history.png"
            plt.savefig(history_fig_path)
            mlflow.log_artifact(history_fig_path)
            
            # Check if training was successful (no NaN)
            if np.isnan(history.history['loss']).any():
                print("Warning: NaN values occurred during training. Using fallback model.")
                # Create a simpler fallback model
                simple_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    learning_rate=0.05,
                    max_depth=3,
                    n_estimators=100
                )
                simple_model.fit(X_train, y_train)
                y_pred = simple_model.predict(X_test)
                self.best_model = simple_model
                self.best_model_name = "xgboost_fallback"
                model = simple_model
            else:
                # Make predictions and evaluate
                y_pred = model.predict(X_test).flatten()
            
            # Compute metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            print(f"MLP RMSE: {rmse:.2f}")
            print(f"MLP MAE: {mae:.2f}")
            print(f"MLP R²: {r2:.2f}")
            
            # Log model
            mlflow.keras.log_model(model, "mlp_model")
            
            # Update best model if this one is better
            if rmse < self.best_score:
                self.best_score = rmse
                self.best_model = model
                self.best_model_name = "mlp"
            
            return model
    
    def save_model(self, model, model_type, model_dir="models"):
        """
        Save the trained model to disk
        
        Args:
            model: Trained model
            model_type (str): Type of model ('xgboost', 'mlp', etc.)
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
        elif model_type == "mlp":
            model.save(model_path)
            return model_path
        else:
            joblib.dump(model, f"{model_path}.pkl")
            return f"{model_path}.pkl"
    
    def load_model(self, model_path, model_type):
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            model_type (str): Type of model ('xgboost', 'mlp', etc.)
            
        Returns:
            Trained model
        """
        if model_type == "xgboost":
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        elif model_type == "mlp":
            from tensorflow.keras.models import load_model
            return load_model(model_path)
        else:
            return joblib.load(model_path)
    
    def get_best_model(self):
        """
        Get the best performing model
        
        Returns:
            tuple: (best_model, best_model_name)
        """
        return self.best_model, self.best_model_name