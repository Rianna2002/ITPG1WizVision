"""
Quick test with 2024-06 as default date
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor

print("=== Testing with 2024-06 as default date ===")

# Load and process data
dp = DataProcessor()
data = dp.load_data("data/RentingOutofFlats2025.csv")
X_train, X_test, y_train, y_test = dp.preprocess_data(data)

# Train XGBoost
trainer = ModelTrainer()
model = trainer.train_xgboost(X_train, y_train, X_test, y_test)

# Create predictor
predictor = RentalPredictor(
    preprocessor=dp,
    model=model,
    model_type="xgboost"
)

# Test different inputs WITHOUT providing a date (should use 2024-06)
test_inputs = [
    {'town': 'ANG MO KIO', 'flat_type': '1 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '5 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'BEDOK', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'CENTRAL', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
]

print("\nTesting different inputs (should get varied predictions):")
for i, input_data in enumerate(test_inputs):
    try:
        prediction = predictor.predict(input_data)
        print(f"{i+1}. {input_data['town']} {input_data['flat_type']}: S${prediction:.2f}")
    except Exception as e:
        print(f"{i+1}. ERROR: {e}")

print("\nDone!")
