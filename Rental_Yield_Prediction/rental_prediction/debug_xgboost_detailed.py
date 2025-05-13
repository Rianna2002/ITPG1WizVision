"""
More targeted debugging to find the exact XGBoost issue
Save as debug_xgboost_detailed.py
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor
import xgboost as xgb

print("=== Detailed XGBoost Debugging ===")

# Load and process data
dp = DataProcessor()
data = dp.load_data("data/RentingOutofFlats2025.csv")
X_train, X_test, y_train, y_test = dp.preprocess_data(data)

# Train XGBoost
trainer = ModelTrainer()
model = trainer.train_xgboost(X_train, y_train, X_test, y_test)

print(f"\n1. Model after training:")
print(f"   Type: {type(model)}")
print(f"   ID: {id(model)}")

# Test the model directly
print(f"\n2. Direct model testing:")
test_samples = X_test[:3]
direct_predictions = model.predict(test_samples)
print(f"   Direct predictions: {direct_predictions}")

# Create predictor
predictor = RentalPredictor(
    preprocessor=dp,
    model=model,
    model_type="xgboost"
)

print(f"\n3. Predictor model info:")
print(f"   Type: {type(predictor.model)}")
print(f"   ID: {id(predictor.model)}")
print(f"   Same object? {predictor.model is model}")

# Test the predictor with the same preprocessed data
print(f"\n4. Testing predictor with preprocessed data:")
# We need to simulate what the predictor does

# Create test inputs
test_inputs = [
    {'town': 'ANG MO KIO', 'flat_type': '1 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'TAMPINES', 'flat_type': '3 ROOM', 'block': '456', 'street_name': 'ANOTHER STREET'},
]

for i, input_data in enumerate(test_inputs):
    print(f"\n   Test {i+1}: {input_data}")
    
    # Process the input manually
    processed = dp.preprocess_input(input_data)
    print(f"   Processed shape: {processed.shape}")
    
    # Convert to proper format
    if hasattr(processed, 'toarray'):
        processed_dense = processed.toarray()
    else:
        processed_dense = processed
    
    print(f"   Dense shape: {processed_dense.shape}")
    
    # Test the model directly with this processed data
    try:
        direct_pred = model.predict(processed_dense)
        print(f"   Direct model prediction: {direct_pred}")
    except Exception as e:
        print(f"   Direct model error: {e}")
    
    # Test through predictor
    try:
        predictor_pred = predictor.predict(input_data)
        print(f"   Predictor prediction: {predictor_pred}")
    except Exception as e:
        print(f"   Predictor error: {e}")

# Let's also test if the issue is with the preprocessing
print(f"\n5. Testing preprocessing consistency:")
test_input = {'town': 'ANG MO KIO', 'flat_type': '1 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'}

# Process twice
processed1 = dp.preprocess_input(test_input)
processed2 = dp.preprocess_input(test_input)

print(f"   First processing: shape={processed1.shape}")
print(f"   Second processing: shape={processed2.shape}")
print(f"   Are they equal? {np.array_equal(processed1.toarray(), processed2.toarray())}")

# Test model on both
pred1 = model.predict(processed1.toarray())
pred2 = model.predict(processed2.toarray())
print(f"   Prediction 1: {pred1}")
print(f"   Prediction 2: {pred2}")
print(f"   Are predictions equal? {np.array_equal(pred1, pred2)}")

print("\nDone!")
