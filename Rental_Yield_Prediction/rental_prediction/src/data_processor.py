import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    Enhanced DataProcessor with separate preprocessing for different models
    """
    
    def __init__(self):
        self.preprocessor_xgboost = None  # Without temporal features
        self.preprocessor_temporal = None  # With temporal features
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_categories = {}
        self.training_date_range = None
        
    def load_data(self, file_path):
        """Load rental data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def standardize_flat_type(self, flat_type):
        """Standardize flat type format"""
        if isinstance(flat_type, str):
            flat_type = flat_type.replace('-', ' ').upper()
            flat_type_map = {
                '1 ROOM': '1 ROOM',
                '2 ROOM': '2 ROOM', 
                '3 ROOM': '3 ROOM',
                '4 ROOM': '4 ROOM',
                '5 ROOM': '5 ROOM',
                'EXECUTIVE': 'EXECUTIVE',
                'MULTI GENERATION': 'MULTI-GENERATION',
                'MULTI-GENERATION': 'MULTI-GENERATION'
            }
            return flat_type_map.get(flat_type, flat_type)
        return flat_type
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """
        Preprocess data with separate pipelines for XGBoost and temporal models
        """
        data = df.copy()
        
        # Extract temporal features
        data['rent_approval_date'] = pd.to_datetime(data['rent_approval_date'], 
                                                format='%Y-%m', errors='coerce')
        data['month'] = data['rent_approval_date'].dt.month
        data['year'] = data['rent_approval_date'].dt.year
        
        # Store training date range
        self.training_date_range = {
            'min_year': data['year'].min(),
            'max_year': data['year'].max(),
            'min_month': data['month'].min(),
            'max_month': data['month'].max()
        }
        
        # Print year distribution
        year_counts = data['year'].value_counts().sort_index()
        print(f"[DEBUG] Year distribution in training data:")
        for year, count in year_counts.items():
            print(f"  {year}: {count} samples ({count/len(data)*100:.1f}%)")
        
        # Drop date column
        data = data.drop('rent_approval_date', axis=1)
        
        # Standardize flat types
        data['flat_type'] = data['flat_type'].apply(self.standardize_flat_type)
        
        # Define features and target
        X = data.drop('monthly_rent', axis=1)
        y = data['monthly_rent']
        
        # Store training categories
        self.training_categories = {
            'town': sorted(X['town'].unique()),
            'flat_type': sorted(X['flat_type'].unique()),
            'block': sorted(X['block'].unique()),
            'street_name': sorted(X['street_name'].unique())
        }
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # === Create XGBoost Preprocessor (WITHOUT temporal features) ===
        X_train_xgb = self.X_train.drop(['month', 'year'], axis=1)
        X_test_xgb = self.X_test.drop(['month', 'year'], axis=1)
        
        categorical_cols_xgb = X_train_xgb.select_dtypes(include=['object']).columns.tolist()
        numerical_cols_xgb = X_train_xgb.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"[DEBUG] XGBoost - Categorical columns: {categorical_cols_xgb}")
        print(f"[DEBUG] XGBoost - Numerical columns: {numerical_cols_xgb}")
        
        categorical_transformer_xgb = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer_xgb = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor_xgboost = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer_xgb, categorical_cols_xgb),
                ('num', numerical_transformer_xgb, numerical_cols_xgb)
            ])
        
        # Fit and transform XGBoost data
        X_train_xgb_processed = self.preprocessor_xgboost.fit_transform(X_train_xgb)
        X_test_xgb_processed = self.preprocessor_xgboost.transform(X_test_xgb)
        
        # === Create Temporal Preprocessor (WITH temporal features) ===
        categorical_cols_temp = self.X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols_temp = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"[DEBUG] Temporal - Categorical columns: {categorical_cols_temp}")
        print(f"[DEBUG] Temporal - Numerical columns: {numerical_cols_temp}")
        
        categorical_transformer_temp = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer_temp = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor_temporal = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer_temp, categorical_cols_temp),
                ('num', numerical_transformer_temp, numerical_cols_temp)
            ])
        
        # Fit and transform temporal data
        X_train_temp_processed = self.preprocessor_temporal.fit_transform(self.X_train)
        X_test_temp_processed = self.preprocessor_temporal.transform(self.X_test)
        
        print(f"XGBoost training set size: {X_train_xgb_processed.shape[0]} samples, {X_train_xgb_processed.shape[1]} features")
        print(f"Temporal training set size: {X_train_temp_processed.shape[0]} samples, {X_train_temp_processed.shape[1]} features")
        
        # Return both preprocessed datasets
        return {
            'xgboost': (X_train_xgb_processed, X_test_xgb_processed, self.y_train, self.y_test),
            'temporal': (X_train_temp_processed, X_test_temp_processed, self.y_train, self.y_test)
        }
    
    def preprocess_input(self, input_data, model_type='xgboost'):
        """
        Preprocess user input for prediction
        
        Args:
            input_data (dict): Dictionary containing user input
            model_type (str): 'xgboost' or 'temporal'
        """
        print(f"[DEBUG] Preprocessing input for {model_type}: {input_data}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Standardize flat type
        if 'flat_type' in input_df.columns:
            input_df['flat_type'] = input_df['flat_type'].apply(self.standardize_flat_type)
        
        # Handle temporal features
        if 'rent_approval_date' in input_df.columns:
            input_df['rent_approval_date'] = pd.to_datetime(input_df['rent_approval_date'])
            input_df['month'] = input_df['rent_approval_date'].dt.month
            input_df['year'] = input_df['rent_approval_date'].dt.year
            input_df = input_df.drop('rent_approval_date', axis=1)
        else:
            # For XGBoost, we don't need temporal features
            if model_type == 'temporal':
                # For temporal models, use a representative date
                input_df['year'] = 2024
                input_df['month'] = 6
                print(f"[INFO] Using default date for temporal model: 2024-06")
        
        # For XGBoost, remove temporal features
        if model_type == 'xgboost':
            # Remove temporal columns if they exist
            temporal_cols = ['month', 'year']
            for col in temporal_cols:
                if col in input_df.columns:
                    input_df = input_df.drop(col, axis=1)
            print(f"[INFO] Removed temporal features for XGBoost")
        
        # Ensure required columns are present
        if model_type == 'xgboost':
            required_cols = ['town', 'block', 'street_name', 'flat_type']
        else:
            required_cols = ['town', 'block', 'street_name', 'flat_type', 'month', 'year']
        
        for col in required_cols:
            if col not in input_df.columns:
                print(f"[WARNING] Missing column {col}, adding default value")
                if col in ['month', 'year']:
                    input_df[col] = 6 if col == 'month' else 2024
                else:
                    input_df[col] = ''
        
        # Select appropriate preprocessor
        preprocessor = self.preprocessor_xgboost if model_type == 'xgboost' else self.preprocessor_temporal
        
        # Get expected column order
        if hasattr(preprocessor, 'transformers_'):
            column_order = []
            for name, transformer, cols in preprocessor.transformers_:
                column_order.extend(cols)
            
            # Reorder columns
            input_df = input_df.reindex(columns=column_order, fill_value='')
        
        # Transform input
        processed_input = preprocessor.transform(input_df)
        
        print(f"[DEBUG] {model_type} processed input shape: {processed_input.shape}")
        
        return processed_input