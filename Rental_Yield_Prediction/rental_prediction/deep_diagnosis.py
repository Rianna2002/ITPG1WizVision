"""
Comprehensive model diagnostics
Save as deep_model_diagnosis.py
"""

import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import RentalPredictor
import matplotlib.pyplot as plt

print("=== Deep Model Diagnosis ===")

# Load and process data
dp = DataProcessor()
data = dp.load_data("data/RentingOutofFlats2025.csv")
X_train, X_test, y_train, y_test = dp.preprocess_data(data)

# Train XGBoost
trainer = ModelTrainer()
model = trainer.train_xgboost(X_train, y_train, X_test, y_test)

print(f"\n1. Model diagnostics:")
print(f"   Model type: {type(model)}")
print(f"   Number of features: {model.n_features_in_}")
print(f"   Model objective: {model.objective}")

# Test on original training/test data
print(f"\n2. Testing on original data:")
train_pred_sample = model.predict(X_train[:5])
test_pred_sample = model.predict(X_test[:5])
print(f"   Train predictions: {train_pred_sample}")
print(f"   Test predictions: {test_pred_sample}")
print(f"   Train actuals: {y_train.iloc[:5].values}")
print(f"   Test actuals: {y_test.iloc[:5].values}")

# Feature importance
print(f"\n3. Feature importance (top 20):")
importance = model.feature_importances_
top_20_idx = np.argsort(importance)[-20:]
print(f"   Top 20 feature indices: {top_20_idx}")
print(f"   Top 20 importance values: {importance[top_20_idx]}")

# Manual feature analysis
print(f"\n4. Manual categorical feature test:")

# Create test data manually
test_cases = [
    # Different towns, same everything else
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'BEDOK', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'CENTRAL', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    
    # Different flat types, same everything else
    {'town': 'ANG MO KIO', 'flat_type': '1 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '3 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
    {'town': 'ANG MO KIO', 'flat_type': '5 ROOM', 'block': '123', 'street_name': 'SAMPLE STREET'},
]

# Create predictor
predictor = RentalPredictor(
    preprocessor=dp,
    model=model,
    model_type="xgboost"
)

# Process each test case manually
for i, test_case in enumerate(test_cases):
    print(f"\n   Test case {i+1}: {test_case}")
    
    # Process manually
    processed = dp.preprocess_input(test_case)
    
    # Convert to dense for analysis
    if hasattr(processed, 'toarray'):
        dense = processed.toarray()
    else:
        dense = processed
    
    # Find non-zero features
    non_zero_idx = np.nonzero(dense)[1]
    non_zero_vals = dense[0, non_zero_idx]
    
    print(f"   Non-zero indices: {non_zero_idx[:10]}...")
    print(f"   Non-zero values: {non_zero_vals[:10]}...")
    
    # Direct model prediction
    pred = model.predict(dense)
    print(f"   Direct prediction: {pred[0]:.2f}")
    
    # Check if importance overlaps with non-zero features
    overlap = np.intersect1d(non_zero_idx, top_20_idx)
    print(f"   Features in top 20 importance: {overlap}")

# Let's also check what the model learned about specific categories
print(f"\n5. Deeper feature analysis:")

# Get the feature names if possible
try:
    if hasattr(dp.preprocessor, 'get_feature_names_out'):
        feature_names = dp.preprocessor.get_feature_names_out()
        print(f"   Total features: {len(feature_names)}")
        
        # Check town features
        town_features = [i for i, name in enumerate(feature_names) if 'town' in str(name).lower()]
        print(f"   Town feature indices: {town_features[:10]}...")
        
        # Check flat type features
        flat_features = [i for i, name in enumerate(feature_names) if 'flat_type' in str(name).lower()]
        print(f"   Flat type feature indices: {flat_features[:10]}...")
        
        # Check importance of these features
        town_importance = importance[town_features]
        flat_importance = importance[flat_features]
        print(f"   Town feature importance (avg): {np.mean(town_importance):.6f}")
        print(f"   Flat type feature importance (avg): {np.mean(flat_importance):.6f}")
        
except Exception as e:
    print(f"   Error getting feature names: {e}")

# Test with completely different data
print(f"\n6. Testing with synthetic data:")

# Create synthetic test data
synthetic_tests = []
for town_idx in [0, 1, 7]:  # ANG MO KIO, BEDOK, CENTRAL
    for flat_idx in [3371, 3373, 3375]:  # 1 ROOM, 3 ROOM, 5 ROOM
        test_array = np.zeros((1, X_train.shape[1]))
        test_array[0, town_idx] = 1.0
        test_array[0, flat_idx] = 1.0
        test_array[0, 138] = 1.0  # block feature
        test_array[0, 3377] = -0.05977247  # month
        test_array[0, 3378] = 1.16338876   # year
        
        pred = model.predict(test_array)
        synthetic_tests.append((town_idx, flat_idx, pred[0]))
        print(f"   Town={town_idx}, Flat={flat_idx}: {pred[0]:.2f}")

print(f"\n7. Analysis summary:")
unique_preds = set([p[2] for p in synthetic_tests])
print(f"   Number of unique predictions: {len(unique_preds)}")
print(f"   Unique values: {sorted(unique_preds)}")

if len(unique_preds) == 1:
    print("   ❌ MODEL IS NOT RESPONDING TO CATEGORICAL FEATURES!")
    print("   🔍 Possible issues:")
    print("      - All categorical features have zero importance")
    print("      - Model is completely dominated by temporal features")
    print("      - Training data preprocessing issue")
    print("      - Model is broken/corrupted")
else:
    print("   ✅ Model does respond to different features")

print("\nDone!")
