import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Enhanced DataProcessor with 80% of median rule outlier filtering by town and flat_type grouping
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
        self.outlier_summary = None
        self.original_data_stats = None
        self.cleaned_data_stats = None
        
    def load_data(self, file_path):
        """Load rental data from CSV or Excel file"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Auto-detect column structure
            df = self._standardize_column_names(df)
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _standardize_column_names(self, df):
        """Standardize column names based on common patterns"""
        # Expected column names for rental data
        expected_columns = ['month', 'town', 'block', 'street_name', 'flat_type', 'rental_price']
        
        if len(df.columns) == 6:
            df.columns = expected_columns
            print(f"Standardized column names to: {list(df.columns)}")
        elif 'rent_approval_date' in df.columns and 'monthly_rent' in df.columns:
            # Already standardized format
            pass
        else:
            print(f"Current columns: {list(df.columns)}")
            print("Please ensure your data has columns: month, town, block, street_name, flat_type, rental_price")
            
        return df
    
    def apply_80_percent_median_rule_by_town_flattype(self, df, price_column='rental_price', town_column='town', flat_type_column='flat_type'):
        """
        Apply 80% of MEDIAN rule outlier filtering by town and flat_type grouping
        Remove rentals < 80% of median for each (town, flat_type) combination
        """
        print("\n" + "="*70)
        print("APPLYING 80% OF MEDIAN RULE OUTLIER FILTERING")
        print("GROUP BY: TOWN → FLAT_TYPE")
        print("="*70)
        
        # Store original data statistics
        self.original_data_stats = {
            'total_records': len(df),
            'mean_price': df[price_column].mean(),
            'median_price': df[price_column].median(),
            'min_price': df[price_column].min(),
            'max_price': df[price_column].max(),
            'std_price': df[price_column].std()
        }
        
        print(f"Original dataset size: {len(df):,}")
        
        clean_data_list = []
        outlier_summary = []
        
        # Calculate MEDIAN for each (town, flat_type) combination
        group_medians = df.groupby([town_column, flat_type_column])[price_column].median()
        group_means = df.groupby([town_column, flat_type_column])[price_column].mean()
        group_counts = df.groupby([town_column, flat_type_column])[price_column].count()
        
        print(f"\nProcessing {len(group_medians)} unique (town, flat_type) combinations...")
        print(f"Sample combinations:")
        for i, (key, median_price) in enumerate(group_medians.head(10).items()):
            town, flat_type = key
            threshold = median_price * 0.8
            count = group_counts[key]
            print(f"  {town} - {flat_type}: {count} records, Median ${median_price:.0f} → 80% Threshold ${threshold:.0f}")
        
        # Process each town-flat_type combination
        total_outliers_removed = 0
        
        for (town, flat_type), median_price in group_medians.items():
            # Get subset for this specific (town, flat_type) combination
            subset = df[(df[town_column] == town) & (df[flat_type_column] == flat_type)].copy()
            
            if len(subset) == 0:
                continue
            
            # Calculate 80% threshold based on MEDIAN for this combination
            mean_price = group_means[(town, flat_type)]
            threshold_80pct_median = median_price * 0.8
            
            # Keep only prices >= 80% of median
            clean_subset = subset[subset[price_column] >= threshold_80pct_median].copy()
            outliers_removed = len(subset) - len(clean_subset)
            total_outliers_removed += outliers_removed
            
            # Track outlier removal summary
            outlier_summary.append({
                'town': town,
                'flat_type': flat_type,
                'original_count': len(subset),
                'clean_count': len(clean_subset),
                'outliers_removed': outliers_removed,
                'outlier_percentage': (outliers_removed / len(subset)) * 100 if len(subset) > 0 else 0,
                'median_price_original': median_price,
                'mean_price_original': mean_price,
                'threshold_80pct_median': threshold_80pct_median,
                'mean_price_clean': clean_subset[price_column].mean() if len(clean_subset) > 0 else 0,
                'median_price_clean': clean_subset[price_column].median() if len(clean_subset) > 0 else 0,
                'min_price_original': subset[price_column].min(),
                'min_price_clean': clean_subset[price_column].min() if len(clean_subset) > 0 else 0,
                'max_price_original': subset[price_column].max(),
                'max_price_clean': clean_subset[price_column].max() if len(clean_subset) > 0 else 0,
                'std_price_original': subset[price_column].std(),
                'std_price_clean': clean_subset[price_column].std() if len(clean_subset) > 0 else 0
            })
            
            if len(clean_subset) > 0:
                clean_data_list.append(clean_subset)
        
        # Combine all clean data
        df_clean = pd.concat(clean_data_list, ignore_index=True)
        
        # Store cleaned data statistics
        self.cleaned_data_stats = {
            'total_records': len(df_clean),
            'mean_price': df_clean[price_column].mean(),
            'median_price': df_clean[price_column].median(),
            'min_price': df_clean[price_column].min(),
            'max_price': df_clean[price_column].max(),
            'std_price': df_clean[price_column].std()
        }
        
        print(f"\nFinal dataset size: {len(df_clean):,}")
        print(f"Total outliers removed: {total_outliers_removed:,}")
        print(f"Percentage of data retained: {(len(df_clean) / len(df)) * 100:.2f}%")
        
        # Create summary DataFrame
        self.outlier_summary = pd.DataFrame(outlier_summary)
        
        # Display summary by town and flat type
        print(f"\nOutlier Removal Summary (80% of Median Rule by Town-FlatType):")
        print("Top 20 combinations with most outliers removed:")
        
        if len(self.outlier_summary) > 0:
            # Show top combinations with most outliers
            top_outliers = self.outlier_summary.nlargest(20, 'outliers_removed')
            display_cols = ['town', 'flat_type', 'original_count', 'clean_count', 'outliers_removed', 
                           'outlier_percentage', 'median_price_original', 'threshold_80pct_median']
            summary_display = top_outliers[display_cols].round(1)
            print(summary_display.to_string(index=False))
            
            # Summary by flat type
            print(f"\nSummary by Flat Type:")
            flat_type_summary = self.outlier_summary.groupby('flat_type').agg({
                'original_count': 'sum',
                'clean_count': 'sum',
                'outliers_removed': 'sum'
            }).reset_index()
            flat_type_summary['outlier_percentage'] = (
                flat_type_summary['outliers_removed'] / flat_type_summary['original_count'] * 100
            )
            print(flat_type_summary.round(1).to_string(index=False))
            
            # Summary by town
            print(f"\nSummary by Town (Top 10):")
            town_summary = self.outlier_summary.groupby('town').agg({
                'original_count': 'sum',
                'clean_count': 'sum',
                'outliers_removed': 'sum'
            }).reset_index()
            town_summary['outlier_percentage'] = (
                town_summary['outliers_removed'] / town_summary['original_count'] * 100
            )
            town_summary = town_summary.nlargest(10, 'outliers_removed')
            print(town_summary.round(1).to_string(index=False))
        
        # Validate flat type logic
        self._validate_flat_type_logic(df_clean, price_column, flat_type_column)
        
        return df_clean, self.outlier_summary
    
    def _validate_flat_type_logic(self, df, price_column='rental_price', flat_type_column='flat_type'):
        """Validate that flat type price relationships make logical sense"""
        print(f"\n" + "="*70)
        print("FLAT TYPE PRICE VALIDATION (AFTER 80% MEDIAN FILTERING BY TOWN-FLATTYPE)")
        print("="*70)
        
        flat_stats = df.groupby(flat_type_column)[price_column].agg(['min', 'median', 'mean', 'max', 'count']).round(0)
        flat_stats = flat_stats.sort_values('median')
        
        print("Overall Flat Type Price Statistics (sorted by MEDIAN price):")
        print(flat_stats.to_string())
        
        # Check logical order based on median
        flat_order = ['1-ROOM', '2-ROOM', '3-ROOM', '4-ROOM', '5-ROOM', 'EXECUTIVE']
        available_types = [ft for ft in flat_order if ft in flat_stats.index]
        
        print(f"\nLogical Flat Type Order Check (Based on Median):")
        prev_median = 0
        logical_order = True
        
        for flat_type in available_types:
            current_median = flat_stats.loc[flat_type, 'median']
            current_mean = flat_stats.loc[flat_type, 'mean']
            current_min = flat_stats.loc[flat_type, 'min']
            max_price = flat_stats.loc[flat_type, 'max']
            count = flat_stats.loc[flat_type, 'count']
            
            median_ok = current_median >= prev_median
            
            if not median_ok:
                print(f"❌ {flat_type}: Median ${current_median:.0f} (LOWER than previous!) - {count} records")
                logical_order = False
            else:
                print(f"✅ {flat_type}: Median ${current_median:.0f}, Mean ${current_mean:.0f}, Min ${current_min:.0f}, Max ${max_price:.0f} - {count} records")
            
            prev_median = current_median
        
        if logical_order:
            print(f"\n✅ Flat type price order is LOGICAL (based on median)!")
        else:
            print(f"\n❌ Flat type price order has some ISSUES!")
        
        return logical_order
    
    def clean_and_prepare_data(self, file_path=None, df=None, apply_outlier_filter=True, export_cleaned_data=False):
        """
        Complete data cleaning pipeline with 80% of median rule by town-flattype grouping
        """
        print("="*80)
        print("RENTAL DATA CLEANING AND PREPARATION PIPELINE")
        print("80% OF MEDIAN RULE BY TOWN → FLAT_TYPE GROUPING")
        print("="*80)
        
        # Load data if file path provided
        if file_path:
            df = self.load_data(file_path)
            if df is None:
                return None, None
        elif df is None:
            print("Error: No data provided")
            return None, None
        
        # Basic data cleaning
        df = self._basic_data_cleaning(df)
        
        # Apply outlier filtering if requested
        cleaned_df = df.copy()
        outlier_summary = None
        
        if apply_outlier_filter:
            price_col = 'rental_price' if 'rental_price' in df.columns else 'monthly_rent'
            town_col = 'town'
            flat_type_col = 'flat_type'
            
            cleaned_df, outlier_summary = self.apply_80_percent_median_rule_by_town_flattype(
                df, price_col, town_col, flat_type_col
            )
        
        # Standardize for training
        cleaned_df = self._prepare_for_model_training(cleaned_df)
        
        # Export if requested
        if export_cleaned_data and file_path:
            self._export_cleaned_data(cleaned_df, outlier_summary, file_path)
        
        print(f"\n✅ Data cleaning pipeline completed successfully!")
        print(f"   Original records: {len(df):,}")
        print(f"   Cleaned records: {len(cleaned_df):,}")
        print(f"   Retention rate: {len(cleaned_df)/len(df)*100:.1f}%")
        
        return cleaned_df, outlier_summary
    
    def _basic_data_cleaning(self, df):
        """Basic data cleaning and validation"""
        print(f"\nPerforming basic data cleaning...")
        
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        if len(df) < original_count:
            print(f"  Removed {original_count - len(df)} duplicate records")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  Missing values found:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"    {col}: {count} missing values")
            
            # Drop rows with missing critical values
            critical_cols = ['town', 'flat_type']
            price_col = 'rental_price' if 'rental_price' in df.columns else 'monthly_rent'
            if price_col in df.columns:
                critical_cols.append(price_col)
            
            df = df.dropna(subset=critical_cols)
            print(f"  Dropped rows with missing critical values")
        
        # Ensure price column is numeric
        price_col = 'rental_price' if 'rental_price' in df.columns else 'monthly_rent'
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df = df.dropna(subset=[price_col])
        
        # Standardize flat types
        if 'flat_type' in df.columns:
            df['flat_type'] = df['flat_type'].apply(self.standardize_flat_type)
        
        print(f"  Basic cleaning completed. Records: {len(df):,}")
        return df
    
    def _prepare_for_model_training(self, df):
        """Prepare data for model training"""
        print(f"\nPreparing data for model training...")
        
        # Handle date column
        if 'month' in df.columns:
            df['rent_approval_date'] = df['month']
            if 'rental_price' in df.columns:
                df['monthly_rent'] = df['rental_price']
            
            # Drop original columns
            df = df.drop(['month'], axis=1)
            if 'rental_price' in df.columns:
                df = df.drop(['rental_price'], axis=1)
        
        print(f"  Data prepared for training. Final columns: {list(df.columns)}")
        return df
    
    def _export_cleaned_data(self, cleaned_df, outlier_summary, original_file_path):
        """Export cleaned data to Excel"""
        try:
            original_path = Path(original_file_path)
            output_path = original_path.parent / f"{original_path.stem}_cleaned_80pct_median_town_flattype.xlsx"
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                cleaned_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                
                if outlier_summary is not None:
                    outlier_summary.to_excel(writer, sheet_name='Outlier_Summary_TownFlatType', index=False)
                    
                    # Summary by flat type
                    flat_type_summary = outlier_summary.groupby('flat_type').agg({
                        'original_count': 'sum',
                        'clean_count': 'sum',
                        'outliers_removed': 'sum'
                    }).reset_index()
                    flat_type_summary['outlier_percentage'] = (
                        flat_type_summary['outliers_removed'] / flat_type_summary['original_count'] * 100
                    )
                    flat_type_summary.to_excel(writer, sheet_name='Summary_by_FlatType', index=False)
                    
                    # Summary by town
                    town_summary = outlier_summary.groupby('town').agg({
                        'original_count': 'sum',
                        'clean_count': 'sum',
                        'outliers_removed': 'sum'
                    }).reset_index()
                    town_summary['outlier_percentage'] = (
                        town_summary['outliers_removed'] / town_summary['original_count'] * 100
                    )
                    town_summary = town_summary.sort_values('outliers_removed', ascending=False)
                    town_summary.to_excel(writer, sheet_name='Summary_by_Town', index=False)
                
                if self.original_data_stats and self.cleaned_data_stats:
                    comparison_df = pd.DataFrame({
                        'Metric': ['Total Records', 'Mean Price', 'Median Price', 'Min Price', 'Max Price', 'Std Dev'],
                        'Original': [
                            self.original_data_stats['total_records'],
                            round(self.original_data_stats['mean_price'], 0),
                            round(self.original_data_stats['median_price'], 0),
                            self.original_data_stats['min_price'],
                            self.original_data_stats['max_price'],
                            round(self.original_data_stats['std_price'], 0)
                        ],
                        'Cleaned_80pct_Median_TownFlatType': [
                            self.cleaned_data_stats['total_records'],
                            round(self.cleaned_data_stats['mean_price'], 0),
                            round(self.cleaned_data_stats['median_price'], 0),
                            self.cleaned_data_stats['min_price'],
                            self.cleaned_data_stats['max_price'],
                            round(self.cleaned_data_stats['std_price'], 0)
                        ]
                    })
                    comparison_df.to_excel(writer, sheet_name='Before_After_Comparison', index=False)
            
            print(f"  ✅ Cleaned data exported to: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error exporting cleaned data: {e}")
    
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
                'EXECUTIVE': 'EXECUTIVE'
            }
            return flat_type_map.get(flat_type, flat_type)
        return flat_type
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess data for both XGBoost and temporal models"""
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
        
        # Create XGBoost Preprocessor (WITHOUT temporal features)
        X_train_xgb = self.X_train.drop(['month', 'year'], axis=1)
        X_test_xgb = self.X_test.drop(['month', 'year'], axis=1)
        
        categorical_cols_xgb = X_train_xgb.select_dtypes(include=['object']).columns.tolist()
        numerical_cols_xgb = X_train_xgb.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
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
        
        # Create Temporal Preprocessor (WITH temporal features)
        categorical_cols_temp = self.X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols_temp = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
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
        
        print(f"XGBoost training set: {X_train_xgb_processed.shape[0]} samples, {X_train_xgb_processed.shape[1]} features")
        print(f"Temporal training set: {X_train_temp_processed.shape[0]} samples, {X_train_temp_processed.shape[1]} features")
        
        return {
            'xgboost': (X_train_xgb_processed, X_test_xgb_processed, self.y_train, self.y_test),
            'temporal': (X_train_temp_processed, X_test_temp_processed, self.y_train, self.y_test)
        }
    
    def preprocess_input(self, input_data, model_type='xgboost'):
        """Preprocess user input for prediction"""
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
        elif model_type == 'temporal' and ('month' not in input_df.columns or 'year' not in input_df.columns):
            # For temporal models, use default date if not provided
            input_df['year'] = 2024
            input_df['month'] = 6
        
        # For XGBoost, remove temporal features
        if model_type == 'xgboost':
            temporal_cols = ['month', 'year']
            for col in temporal_cols:
                if col in input_df.columns:
                    input_df = input_df.drop(col, axis=1)
        
        # Select appropriate preprocessor
        preprocessor = self.preprocessor_xgboost if model_type == 'xgboost' else self.preprocessor_temporal
        
        # Transform input
        processed_input = preprocessor.transform(input_df)
        
        return processed_input
    
    def get_cleaning_summary(self):
        """Get summary of data cleaning process"""
        if self.outlier_summary is None:
            return "No outlier filtering applied"
        
        summary = {
            'original_stats': self.original_data_stats,
            'cleaned_stats': self.cleaned_data_stats,
            'outlier_summary': self.outlier_summary,
            'retention_rate': (self.cleaned_data_stats['total_records'] / 
                             self.original_data_stats['total_records']) * 100,
            'filtering_method': '80% of Median Rule (Group by Town → Flat Type)'
        }
        
        return summary

# Example usage for clients
def example_usage():
    """Example of how clients can use the enhanced DataProcessor with 80% of median rule by town-flattype"""
    
    # Initialize processor
    processor = DataProcessor()
    
    # Method 1: Complete pipeline (Load → Clean → Export → Prepare for Training)
    cleaned_df, outlier_summary = processor.clean_and_prepare_data(
        file_path="data/RentingOutofFlats2025.csv",
        apply_outlier_filter=True,
        export_cleaned_data=True
    )
    
    # Method 2: Step-by-step approach with 80% of median rule by town-flattype
    # df = processor.load_data("data/RentingOutofFlats2025.csv")
    # cleaned_df, outlier_summary = processor.apply_80_percent_median_rule_by_town_flattype(df)
    
    # Now you can train models with cleaned data
    if cleaned_df is not None:
        processed_data = processor.preprocess_data(cleaned_df)
        
        # Get XGBoost data
        X_train_xgb, X_test_xgb, y_train, y_test = processed_data['xgboost']
        
        # Get temporal model data
        X_train_temp, X_test_temp, y_train, y_test = processed_data['temporal']
        
        print(f"Ready for model training!")
        print(f"XGBoost data shape: {X_train_xgb.shape}")
        print(f"Temporal data shape: {X_train_temp.shape}")
        
        # Get cleaning summary
        summary = processor.get_cleaning_summary()
        print(f"Filtering method: {summary['filtering_method']}")
        print(f"Data retention rate: {summary['retention_rate']:.1f}%")

if __name__ == "__main__":
    example_usage()