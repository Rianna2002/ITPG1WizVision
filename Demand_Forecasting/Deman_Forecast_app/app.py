# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    xgb_model = joblib.load("best_xgb_model.pkl")
    lstm_model = load_model("best_lstm_model.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    arima_model = joblib.load("arima_best_model.pkl")
    return xgb_model, lstm_model, arima_model

xgb_model, lstm_model, arima_model = load_models()

# Data Preprocessing Code
@st.cache(allow_output_mutation=True)
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv("merged_hdb_resale_data.csv")
    data['month'] = pd.to_datetime(data['month'], format='%Y-%m')

    # Aggregate data: count the number of transactions per month per town
    transaction_count = data.groupby(['month', 'town']).size().reset_index(name='transaction_count')

    return transaction_count

transaction_count = load_and_preprocess_data()

# Sliding Window Data Preparation
def create_sliding_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Make predictions using the loaded models
@st.cache(allow_output_mutation=True)
def make_predictions():
    # Generate the test data directly (assuming X_test and y_test were prepared in Colab)
    transaction_series = transaction_count['transaction_count'].values
    X, y = create_sliding_windows(transaction_series, 12)

    # Split data into train and test sets (80-20)
    train_size = int(len(X) * 0.8)
    X_train_lstm, X_test_lstm = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # XGBoost Predictions
    xgb_pred = xgb_model.predict(X_test_lstm)

    # LSTM Predictions
    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1)).astype(np.float32)
    lstm_pred = lstm_model.predict(X_test_lstm).flatten()

    # ARIMA Predictions (using the entire aggregate data)
    arima_pred = arima_model.forecast(steps=len(y_test))

    return y_test, xgb_pred, lstm_pred, arima_pred

y_test, xgb_pred, lstm_pred, arima_pred = make_predictions()

# Calculate evaluation metrics
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    return rmse, mae, mape

xgb_rmse, xgb_mae, xgb_mape = evaluate(y_test, xgb_pred)
lstm_rmse, lstm_mae, lstm_mape = evaluate(y_test, lstm_pred)
arima_rmse, arima_mae, arima_mape = evaluate(y_test, arima_pred)

# Display metrics in a table
st.title("Transaction Count Prediction Models")
st.subheader("Model Metrics")
metrics_df = pd.DataFrame({
    "Model": ["XGBoost", "LSTM", "ARIMA"],
    "RMSE": [xgb_rmse, lstm_rmse, arima_rmse],
    "MAE": [xgb_mae, lstm_mae, arima_mae],
    "MAPE (%)": [xgb_mape, lstm_mape, arima_mape]
})
st.dataframe(metrics_df)

# Plot Actual vs Predicted
def plot_predictions(y_true, y_preds, model_names):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, (y_pred, model_name) in enumerate(zip(y_preds, model_names)):
        axs[i].scatter(y_true, y_pred, alpha=0.5, label='Predicted')
        axs[i].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Fit')
        axs[i].set_title(f"Actual vs Predicted - {model_name}")
        axs[i].set_xlabel("Actual Transaction Count")
        axs[i].set_ylabel("Predicted Transaction Count")
        axs[i].legend()
        axs[i].grid(True)
    return fig

st.subheader("Actual vs Predicted Scatter Plots")
fig = plot_predictions(y_test, [xgb_pred, lstm_pred, arima_pred], ["XGBoost", "LSTM", "ARIMA"])
st.pyplot(fig)
