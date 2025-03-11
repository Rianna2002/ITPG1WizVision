import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    Class for preprocessing rental data, handling categorical variables,
    and splitting data into training and testing sets.
    """
    
    def __init__(self):
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path):
        """Load rental data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """
        Preprocess data:
        - Handle date columns
        - Encode categorical variables
        - Scale numerical features
        - Split into train and test sets
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Extract month and year from rent_approval_date
        data['rent_approval_date'] = pd.to_datetime(data['rent_approval_date'], 
                                            format='%Y-%m', 
                                            errors='coerce')
        data['month'] = data['rent_approval_date'].dt.month
        data['year'] = data['rent_approval_date'].dt.year
        
        # Drop the original date column
        data = data.drop('rent_approval_date', axis=1)
        
        # Define features and target
        X = data.drop('monthly_rent', axis=1)
        y = data['monthly_rent']
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, numerical_cols)
            ])
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit and transform training data
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        
        # Transform test data
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"Training set size: {X_train_processed.shape[0]} samples")
        print(f"Testing set size: {X_test_processed.shape[0]} samples")
        
        return X_train_processed, X_test_processed, self.y_train, self.y_test
    
    def preprocess_input(self, input_data):
        """
        Preprocess user input for prediction
        
        Args:
            input_data (dict): Dictionary containing user input
            
        Returns:
            numpy.ndarray: Processed input data for prediction
        """
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Add month and year columns if rent_approval_date is provided
        if 'rent_approval_date' in input_df.columns:
            input_df['rent_approval_date'] = pd.to_datetime(input_df['rent_approval_date'])
            input_df['month'] = input_df['rent_approval_date'].dt.month
            input_df['year'] = input_df['rent_approval_date'].dt.year
            input_df = input_df.drop('rent_approval_date', axis=1)
        else:
            # Use current month and year if not provided
            current_date = pd.Timestamp.now()
            input_df['month'] = current_date.month
            input_df['year'] = current_date.year
        
        # Process input data using the fitted preprocessor
        processed_input = self.preprocessor.transform(input_df)
        
        return processed_input
    
    def get_encoder_feature_names(self):
        """Get feature names after one-hot encoding"""
        categorical_cols = []
        for name, transformer, cols in self.preprocessor.transformers_:
            if name == 'cat':
                categorical_cols.extend(transformer.named_steps['onehot'].get_feature_names_out(cols))
            elif name == 'num':
                categorical_cols.extend(cols)
        return categorical_cols