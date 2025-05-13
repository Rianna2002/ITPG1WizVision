"""
Test with varied temporal features to find values that allow categorical features to work
Save as test_varied_temporal.py
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor

print("=== Testing Varied Temporal Features ===")

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

# Test different year/month combinations
test_combinations = [
    # Try years with good data distribution
    ("2023", "06"),  # Good distribution year
    ("2023", "12"), 
    ("2024", "01"),  # Different month in 2024
    ("2024", "12"),  
    ("2022", "06"),  # Even older year
]

test_cases = [
    {'town': 'ANG MO KIO', 'flat_type': '1 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '5 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'BEDOK', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'CENTRAL', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
]

for year, month in test_combinations:
    print(f"\n{'='*50}")
    print(f"Testing with {year}-{month}")
    print('='*50)
    
    for i, test_case in enumerate(test_cases):
        test_with_date = test_case.copy()
        test_with_date['rent_approval_date'] = f"{year}-{month}"
        
        try:
            prediction = predictor.predict(test_with_date)
            print(f"{i+1}. {test_case['town']} {test_case['flat_type']}: S${prediction:.2f}")
        except Exception as e:
            print(f"{i+1}. ERROR: {e}")
    
    # Check if predictions vary for this date
    predictions = []
    for test_case in test_cases:
        test_with_date = test_case.copy()
        test_with_date['rent_approval_date'] = f"{year}-{month}"
        try:
            pred = predictor.predict(test_with_date)
            predictions.append(pred)
        except:
            pass
    
    unique_preds = len(set(predictions))
    print(f"\nSummary for {year}-{month}:")
    print(f"   Number of unique predictions: {unique_preds}")
    if unique_preds > 1:
        print(f"   ✅ VARIED PREDICTIONS - This date works!")
        print(f"   Range: S${min(predictions):.2f} - S${max(predictions):.2f}")
    else:
        print(f"   ❌ All same prediction: S${predictions[0]:.2f}")

print(f"\n{'='*50}")
print("Conclusion: Use a date that gives varied predictions as the default!")
print('='*50)
