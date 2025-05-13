"""
Test script to check temporal feature impact on predictions
Save as test_temporal_impact.py
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor

print("=== Testing Temporal Feature Impact ===")

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

# Test the same input with different dates
base_input = {
    'town': 'ANG MO KIO',
    'flat_type': '3 ROOM',
    'block': '123',
    'street_name': 'SAMPLE STREET'
}

print(f"\n1. Testing same input with different months/years:")

# Test different temporal combinations
temporal_variations = [
    (1, 2023),   # January 2023
    (6, 2023),   # June 2023
    (12, 2023),  # December 2023
    (1, 2024),   # January 2024
    (6, 2024),   # June 2024
    (12, 2024),  # December 2024
    (1, 2025),   # January 2025
    (2, 2025),   # February 2025 (latest available)
]

for month, year in temporal_variations:
    test_input = base_input.copy()
    test_input['rent_approval_date'] = f"{year}-{month:02d}"
    
    try:
        prediction = predictor.predict(test_input)
        print(f"   {year}-{month:02d}: S${prediction:.2f}")
    except Exception as e:
        print(f"   {year}-{month:02d}: ERROR - {e}")

# Test different input locations with same date
print(f"\n2. Testing different locations with same date:")

test_locations = [
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM'},
    {'town': 'BEDOK', 'flat_type': '3 ROOM'},
    {'town': 'TAMPINES', 'flat_type': '3 ROOM'},
    {'town': 'PASIR RIS', 'flat_type': '3 ROOM'},
    {'town': 'CENTRAL', 'flat_type': '3 ROOM'},
]

fixed_date = "2025-02"  # Use February 2025 as the latest available data

for location in test_locations:
    test_input = location.copy()
    test_input.update({
        'block': '123',
        'street_name': 'SAMPLE STREET',
        'rent_approval_date': fixed_date
    })
    
    try:
        prediction = predictor.predict(test_input)
        print(f"   {location['town']}: S${prediction:.2f}")
    except Exception as e:
        print(f"   {location['town']}: ERROR - {e}")

# Test different flat types with same date
print(f"\n3. Testing different flat types with same date:")

test_flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']

for flat_type in test_flat_types:
    test_input = {
        'town': 'ANG MO KIO',
        'flat_type': flat_type,
        'block': '123',
        'street_name': 'SAMPLE STREET',
        'rent_approval_date': fixed_date
    }
    
    try:
        prediction = predictor.predict(test_input)
        print(f"   {flat_type}: S${prediction:.2f}")
    except Exception as e:
        print(f"   {flat_type}: ERROR - {e}")

print("\nIf all predictions are the same or very similar, the model is overfitting to temporal features!")
print("Consider:")
print("1. Using more diverse temporal features during training")
print("2. Feature engineering to reduce temporal dependencies")
print("3. Adding more non-temporal features")
