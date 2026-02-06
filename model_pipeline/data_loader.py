"""
Data Loader Module
Handles loading, initial validation, and basic exploration of the delivery time dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from . import config
from .utils import setup_logger, validate_input_data, memory_usage, print_section_header


# Setup logger
logger = setup_logger(__name__, config.TRAINING_LOG_PATH)


class DataLoader:
    """
    Class for loading and validating the food delivery dataset.
    
    Attributes:
        data_path: Path to the CSV file
        df: Loaded dataframe
        
    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_data()
        >>> loader.print_data_summary()
    """
    
    def __init__(self, data_path: Path = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data file (uses config.DATA_PATH if None)
        """
        self.data_path = data_path or config.DATA_PATH
        self.df = None
        self.original_shape = None
        
        logger.info(f"DataLoader initialized with path: {self.data_path}")
    
    
    def load_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            verbose: Print loading information
        
        Returns:
            Loaded dataframe
        
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        print_section_header("DATA LOADING")
        
        # Check if file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f" Data file not found at: {self.data_path}")
        
        # Load data
        logger.info(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.original_shape = self.df.shape
        
        if verbose:
            print(f" Data loaded successfully")
            print(f" Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            print(f" Memory usage: {memory_usage(self.df)['total_mb']:.2f} MB")
        
        logger.info(f"Data loaded: {self.df.shape}")
        
        return self.df
    
    
    def validate_data(self) -> bool:
        """
        Validate that data has all required columns and correct types.
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If validation fails
        """
        print_section_header("DATA VALIDATION")
        
        if self.df is None:
            raise ValueError(" No data loaded. Call load_data() first.")
        
        # Check required columns
        all_required_columns = config.ALL_FEATURES + [config.TARGET_COLUMN]
        
        try:
            validate_input_data(self.df, all_required_columns)
            print("All required columns present")
            logger.info("Column validation passed")
        except ValueError as e:
            logger.error(f"Column validation failed: {e}")
            raise
        
        # Check target column
        if config.TARGET_COLUMN not in self.df.columns:
            raise ValueError(f" Target column '{config.TARGET_COLUMN}' not found")
        
        print(f" Target column '{config.TARGET_COLUMN}' found")
        
        # Check data types
        self._validate_data_types()
        
        # Check for completely empty columns
        empty_cols = self.df.columns[self.df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Empty columns found: {empty_cols}")
            print(f"  Warning: Empty columns: {empty_cols}")
        
        print(" Data validation completed\n")
        return True
    
    
    def _validate_data_types(self) -> None:
        """Validate data types of columns."""
        # Numerical columns should be numeric
        for col in config.NUMERICAL_FEATURES:
            if col in self.df.columns:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    logger.warning(f"Column '{col}' is not numeric: {self.df[col].dtype}")
                    print(f"  Warning: '{col}' has non-numeric type: {self.df[col].dtype}")
        
        # Categorical columns should be object/string
        for col in config.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                if self.df[col].dtype not in ['object', 'category']:
                    logger.info(f"Converting '{col}' to string type")
                    self.df[col] = self.df[col].astype(str)
        
        # Target should be numeric
        if not pd.api.types.is_numeric_dtype(self.df[config.TARGET_COLUMN]):
            raise ValueError(f" Target column '{config.TARGET_COLUMN}' must be numeric")
    
    
    def get_missing_value_report(self) -> pd.DataFrame:
        """
        Generate report of missing values.
        
        Returns:
            DataFrame with missing value statistics
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        
        report = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': missing_count.values,
            'Missing_Percentage': missing_pct.values,
            'Data_Type': self.df.dtypes.values
        })
        
        # Sort by missing percentage
        report = report.sort_values('Missing_Percentage', ascending=False)
        report = report[report['Missing_Count'] > 0].reset_index(drop=True)
        
        return report
    
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary with data statistics
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        summary = {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'n_numerical': len([col for col in config.NUMERICAL_FEATURES if col in self.df.columns]),
            'n_categorical': len([col for col in config.CATEGORICAL_FEATURES if col in self.df.columns]),
            'n_missing_total': self.df.isnull().sum().sum(),
            'pct_missing_total': (self.df.isnull().sum().sum() / self.df.size) * 100,
            'memory_mb': memory_usage(self.df)['total_mb'],
            'n_duplicates': self.df.duplicated().sum()
        }
        
        # Target statistics
        if config.TARGET_COLUMN in self.df.columns:
            target_data = self.df[config.TARGET_COLUMN].dropna()
            summary['target_stats'] = {
                'mean': target_data.mean(),
                'median': target_data.median(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max(),
                'missing': self.df[config.TARGET_COLUMN].isnull().sum()
            }
        
        return summary
    
    
    def print_data_summary(self) -> None:
        """Print comprehensive data summary."""
        print_section_header("DATA SUMMARY")
        
        summary = self.get_data_summary()
        
        print(f"📊 Dataset Overview:")
        print(f"   Rows:              {summary['n_rows']:,}")
        print(f"   Columns:           {summary['n_columns']}")
        print(f"   Numerical:         {summary['n_numerical']}")
        print(f"   Categorical:       {summary['n_categorical']}")
        print(f"   Duplicates:        {summary['n_duplicates']}")
        print(f"   Memory Usage:      {summary['memory_mb']:.2f} MB")
        
        print(f"\n🔍 Missing Values:")
        print(f"   Total Missing:     {summary['n_missing_total']:,} ({summary['pct_missing_total']:.2f}%)")
        
        # Missing value report
        missing_report = self.get_missing_value_report()
        if len(missing_report) > 0:
            print(f"\n   Columns with missing values:")
            for _, row in missing_report.iterrows():
                print(f"      {row['Column']:25s}: {int(row['Missing_Count']):4d} ({row['Missing_Percentage']:5.2f}%)")
        else:
            print(" No missing values found")
        
        # Target statistics
        if 'target_stats' in summary:
            stats = summary['target_stats']
            print(f" Target Variable ({config.TARGET_COLUMN}):")
            print(f"   Mean:              {stats['mean']:.2f}")
            print(f"   Median:            {stats['median']:.2f}")
            print(f"   Std Dev:           {stats['std']:.2f}")
            print(f"   Range:             [{stats['min']:.2f}, {stats['max']:.2f}]")
            if stats['missing'] > 0:
                print(f"   Missing:           {stats['missing']}")
        
        print()
    
    
    def get_categorical_summary(self) -> Dict[str, pd.Series]:
        """
        Get summary of categorical variables.
        
        Returns:
            Dictionary with value counts for each categorical column
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        summary = {}
        for col in config.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                summary[col] = self.df[col].value_counts()
        
        return summary
    
    
    def print_categorical_summary(self) -> None:
        """Print summary of categorical variables."""
        print_section_header("CATEGORICAL VARIABLES SUMMARY")
        
        cat_summary = self.get_categorical_summary()
        
        for col, value_counts in cat_summary.items():
            print(f"\n📊 {col}:")
            print(f"   Unique values: {len(value_counts)}")
            print(f"   Distribution:")
            for value, count in value_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"      {str(value):20s}: {count:5d} ({pct:5.2f}%)")
    
    
    def detect_outliers(self, method: str = 'iqr') -> Dict[str, Dict]:
        """
        Detect outliers in numerical columns.
        
        Args:
            method: Method to use ('iqr' or 'zscore')
        
        Returns:
            Dictionary with outlier information for each column
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        outliers = {}
        
        for col in config.NUMERICAL_FEATURES:
            if col not in self.df.columns:
                continue
            
            data = self.df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                outlier_mask = z_scores > 3
                lower_bound = data.mean() - 3 * data.std()
                upper_bound = data.mean() + 3 * data.std()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            n_outliers = outlier_mask.sum()
            
            outliers[col] = {
                'n_outliers': n_outliers,
                'pct_outliers': (n_outliers / len(data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': self.df[self.df[col].notna()].index[outlier_mask].tolist()
            }
        
        return outliers
    
    
    def print_outlier_report(self, method: str = 'iqr') -> None:
        """Print outlier detection report."""
        print_section_header(f"OUTLIER DETECTION ({method.upper()} METHOD)")
        
        outliers = self.detect_outliers(method=method)
        
        for col, info in outliers.items():
            if info['n_outliers'] > 0:
                print(f"\n🔍 {col}:")
                print(f"   Outliers:          {info['n_outliers']} ({info['pct_outliers']:.2f}%)")
                print(f"   Valid range:       [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
            else:
                print(f" {col}: No outliers detected")
        
        print()
    
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded dataframe.
        
        Returns:
            Loaded dataframe
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.df
    
    
    def split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target.
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        X = self.df[config.ALL_FEATURES].copy()
        y = self.df[config.TARGET_COLUMN].copy()
        
        logger.info(f"Data split: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    
    def load_and_validate(self, verbose: bool = True) -> pd.DataFrame:
        """
        Convenience method to load and validate data in one call.
        
        Args:
            verbose: Print information during process
        
        Returns:
            Loaded and validated dataframe
        """
        self.load_data(verbose=verbose)
        self.validate_data()
        
        if verbose:
            self.print_data_summary()
            self.print_categorical_summary()
            self.print_outlier_report()
        
        return self.df


# CONVENIENCE FUNCTIONS

def load_delivery_data(data_path: Path = None, verbose: bool = True) -> pd.DataFrame:
    """
    Quick function to load delivery data.
    
    Args:
        data_path: Path to data file (uses config if None)
        verbose: Print loading information
    
    Returns:
        Loaded dataframe
    
    Example:
        >>> df = load_delivery_data()
    """
    loader = DataLoader(data_path)
    return loader.load_data(verbose=verbose)


def get_train_test_data(test_size: float = None, 
                        random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and split into train/test sets.
    
    Args:
        test_size: Proportion of data for testing (uses config if None)
        random_state: Random seed (uses config if None)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    
    Example:
        >>> X_train, X_test, y_train, y_test = get_train_test_data()
    """
    from sklearn.model_selection import train_test_split
    
    test_size = test_size or config.TEST_SIZE
    random_state = random_state or config.RANDOM_STATE
    
    # Load data
    loader = DataLoader()
    df = loader.load_and_validate(verbose=False)
    
    # Split features and target
    X, y = loader.split_features_target()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} ({(1-test_size)*100:.0f}%/{test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test

# MAIN EXECUTION (for testing)

if __name__ == "__main__":
    """Test the DataLoader class."""
    
    print("TESTING DATA LOADER MODULE")
    
    # Initialize loader
    loader = DataLoader()
    
    # Load and validate data
    df = loader.load_and_validate(verbose=True)
    
    # Display first few rows
    print("\n First 5 rows:")
    print(df.head())
    
    # Split features and target
    X, y = loader.split_features_target()
    print(f"\n  Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    
    print("DATA LOADER TEST COMPLETED")
