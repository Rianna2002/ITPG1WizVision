"""
Run this script separately to test XGBoost training and prediction
Save as test_xgboost.py and run it outside of Streamlit
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor

print("Testing XGBoost independently...")

# Load and process data
dp = DataProcessor()
data = dp.load_data("data/RentingOutofFlats2025.csv")
X_train, X_test, y_train, y_test = dp.preprocess_data(data)

print(f"Training data shape: {X_train.shape}")
print(f"Training data type: {type(X_train)}")

# Train XGBoost
trainer = ModelTrainer()
model = trainer.train_xgboost(X_train, y_train, X_test, y_test)

print(f"Model type: {type(model)}")
print(f"Model parameters: {model.get_params()}")

# Test predictions on training data
train_pred = model.predict(X_test[:5])
print(f"First 5 test predictions: {train_pred}")
print(f"First 5 actual values: {y_test[:5].values}")

# Test with different inputs
test_inputs = [
    {'town': 'ANG MO KIO', 'flat_type': '1 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '5 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'TAMPINES', 'flat_type': '3 ROOM', 'block': '456', 'street_name': 'ANOTHER STREET'},
]

# Test with predictor
predictor = RentalPredictor(
    preprocessor=dp,
    model=model,
    model_type="xgboost"
)

print("\nTesting predictor with different inputs:")
for i, input_data in enumerate(test_inputs):
    try:
        prediction = predictor.predict(input_data)
        print(f"Input {i+1}: {input_data['town']}, {input_data['flat_type']} -> Prediction: S${prediction:.2f}")
    except Exception as e:
        print(f"Input {i+1}: ERROR - {e}")

# Save and reload the model to test persistence
print("\nTesting model save/load...")
model_path, preprocessor_path = predictor.save_model()

# Create new predictor and load model
new_predictor = RentalPredictor()
success = new_predictor.load_from_file(model_path, "xgboost")

if success:
    new_predictor.load_preprocessor(preprocessor_path)
    
    # Test loaded model
    for i, input_data in enumerate(test_inputs[:2]):
        try:
            prediction = new_predictor.predict(input_data)
            print(f"Loaded model - Input {i+1}: Prediction: S${prediction:.2f}")
        except Exception as e:
            print(f"Loaded model - Input {i+1}: ERROR - {e}")
else:
    print("Failed to load model!")

print("\nDone!")
