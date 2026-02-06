"""
Preprocessor Module
Handles data cleaning, missing value imputation, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from . import config
from .utils import setup_logger, save_model, load_model, print_section_header


# Setup logger
logger = setup_logger(__name__, config.TRAINING_LOG_PATH)


class Preprocessor:
    """
    Preprocessor for food delivery time prediction data.
    Handles missing values, encoding, and scaling.
    
    Attributes:
        numerical_imputer: Imputer for numerical features
        categorical_imputer: Imputer for categorical features
        scaler: Scaler for numerical features
        encoders: Dictionary of encoders for categorical features
        feature_names_out: List of feature names after encoding
        is_fitted: Whether the preprocessor has been fitted
    
    Example:
        >>> preprocessor = Preprocessor()
        >>> X_train_clean = preprocessor.fit_transform(X_train)
        >>> X_test_clean = preprocessor.transform(X_test)
    """
    
    def __init__(self):
        """Initialize Preprocessor with empty transformers."""
        # Imputers
        self.numerical_imputer = None
        self.categorical_imputer = None
        
        # Scalers
        self.scaler = None
        
        # Encoders
        self.encoders = {}
        self.encoding_mappings = {}
        
        # Feature names
        self.feature_names_out = []
        self.original_feature_names = []
        
        # Fit status
        self.is_fitted = False
        
        logger.info("Preprocessor initialized")
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'Preprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            y: Training target (not used, for sklearn compatibility)
        
        Returns:
            Self
        """
        print_section_header("FITTING PREPROCESSOR")
        
        self.original_feature_names = X.columns.tolist()
        logger.info(f"Fitting preprocessor on data with shape: {X.shape}")
        
        # 1. Fit imputers
        self._fit_imputers(X)
        
        # 2. Impute missing values
        X_imputed = self._impute_missing_values(X)
        
        # 3. Fit encoders
        self._fit_encoders(X_imputed)
        
        # 4. Encode categorical features
        X_encoded = self._encode_features(X_imputed)
        
        # 5. Fit scaler
        self._fit_scaler(X_encoded)
        
        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")
        print("Preprocessor fitted successfully\n")
        
        return self
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        
        Raises:
            RuntimeError: If preprocessor hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        logger.info(f"Transforming data with shape: {X.shape}")
        
        # 1. Impute missing values
        X_imputed = self._impute_missing_values(X)
        
        # 2. Encode categorical features
        X_encoded = self._encode_features(X_imputed)
        
        # 3. Scale numerical features
        X_scaled = self._scale_features(X_encoded)
        
        logger.info(f"Transformed data shape: {X_scaled.shape}")
        
        return X_scaled
    
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            X: Training features
            y: Training target (not used)
        
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    
    def _fit_imputers(self, X: pd.DataFrame) -> None:
        """Fit imputers for missing values."""
        print("🔧 Fitting imputers...")
        
        # Numerical imputer
        numerical_cols = [col for col in config.NUMERICAL_FEATURES if col in X.columns]
        if numerical_cols:
            self.numerical_imputer = SimpleImputer(
                strategy=config.MISSING_STRATEGY_NUMERICAL,
                add_indicator=False
            )
            self.numerical_imputer.fit(X[numerical_cols])
            logger.info(f"Numerical imputer fitted on {len(numerical_cols)} columns")
            print(f"   ✓ Numerical imputer: {config.MISSING_STRATEGY_NUMERICAL} strategy")
        
        # Categorical imputer
        categorical_cols = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
        if categorical_cols:
            self.categorical_imputer = SimpleImputer(
                strategy=config.MISSING_STRATEGY_CATEGORICAL,
                add_indicator=False
            )
            self.categorical_imputer.fit(X[categorical_cols])
            logger.info(f"Categorical imputer fitted on {len(categorical_cols)} columns")
            print(f"   ✓ Categorical imputer: {config.MISSING_STRATEGY_CATEGORICAL} strategy")
    
    
    def _impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in data."""
        X_imputed = X.copy()
        
        # Numerical imputation
        numerical_cols = [col for col in config.NUMERICAL_FEATURES if col in X.columns]
        if numerical_cols and self.numerical_imputer is not None:
            X_imputed[numerical_cols] = self.numerical_imputer.transform(X[numerical_cols])
        
        # Categorical imputation
        categorical_cols = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
        if categorical_cols and self.categorical_imputer is not None:
            imputed_values = self.categorical_imputer.transform(X[categorical_cols])
            X_imputed[categorical_cols] = imputed_values
        
        return X_imputed
    
    
    def _fit_encoders(self, X: pd.DataFrame) -> None:
        """Fit encoders for categorical features."""
        print("🔧 Fitting encoders...")
        
        categorical_cols = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
        
        if config.ENCODING_STRATEGY == "onehot":
            for col in categorical_cols:
                encoder = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown='ignore',
                    dtype=np.float64
                )
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
                
                # Store categories for reference
                self.encoding_mappings[col] = encoder.categories_[0].tolist()
                
            logger.info(f"OneHot encoders fitted on {len(categorical_cols)} columns")
            print(f"   ✓ OneHot encoding for {len(categorical_cols)} categorical features")
        
        elif config.ENCODING_STRATEGY == "label":
            for col in categorical_cols:
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders[col] = encoder
                
                # Store mapping
                self.encoding_mappings[col] = {
                    label: idx for idx, label in enumerate(encoder.classes_)
                }
            
            logger.info(f"Label encoders fitted on {len(categorical_cols)} columns")
            print(f"   ✓ Label encoding for {len(categorical_cols)} categorical features")
        
        else:
            raise ValueError(f"Unknown encoding strategy: {config.ENCODING_STRATEGY}")
    
    
    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        X_encoded = X.copy()
        
        # Separate numerical and categorical
        numerical_cols = [col for col in config.NUMERICAL_FEATURES if col in X.columns]
        categorical_cols = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
        
        # Keep numerical features as is
        X_numerical = X_encoded[numerical_cols].copy()
        
        # Encode categorical features
        if config.ENCODING_STRATEGY == "onehot":
            encoded_dfs = []
            
            for col in categorical_cols:
                if col in self.encoders:
                    # Transform
                    encoded_array = self.encoders[col].transform(X_encoded[[col]])
                    
                    # Create column names
                    categories = self.encoders[col].categories_[0]
                    col_names = [f"{col}_{cat}" for cat in categories]
                    
                    # Create dataframe
                    encoded_df = pd.DataFrame(
                        encoded_array,
                        columns=col_names,
                        index=X_encoded.index
                    )
                    encoded_dfs.append(encoded_df)
            
            # Combine numerical and encoded categorical
            if encoded_dfs:
                X_encoded = pd.concat([X_numerical] + encoded_dfs, axis=1)
            else:
                X_encoded = X_numerical
        
        elif config.ENCODING_STRATEGY == "label":
            for col in categorical_cols:
                if col in self.encoders:
                    X_encoded[col] = self.encoders[col].transform(X_encoded[col])
            
            # In label encoding, we keep all columns
            X_encoded = X_encoded[numerical_cols + categorical_cols]
        
        # Store feature names
        if not self.feature_names_out:
            self.feature_names_out = X_encoded.columns.tolist()
        
        return X_encoded
    
    
    def _fit_scaler(self, X: pd.DataFrame) -> None:
        """Fit scaler on numerical features."""
        print("🔧 Fitting scaler...")
        
        if config.SCALING_STRATEGY == "standard":
            self.scaler = StandardScaler()
        elif config.SCALING_STRATEGY == "minmax":
            self.scaler = MinMaxScaler()
        elif config.SCALING_STRATEGY == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {config.SCALING_STRATEGY}")
        
        self.scaler.fit(X)
        logger.info(f"Scaler fitted: {config.SCALING_STRATEGY}")
        print(f"   ✓ Scaler: {config.SCALING_STRATEGY}")
    
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using fitted scaler."""
        if self.scaler is None:
            return X
        
        X_scaled_array = self.scaler.transform(X)
        X_scaled = pd.DataFrame(
            X_scaled_array,
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    
    def get_feature_names(self) -> list:
        """
        Get feature names after encoding.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.feature_names_out.copy()
    
    
    def get_encoding_info(self) -> Dict:
        """
        Get information about encodings applied.
        
        Returns:
            Dictionary with encoding information
        """
        return {
            'strategy': config.ENCODING_STRATEGY,
            'mappings': self.encoding_mappings.copy(),
            'n_features_in': len(self.original_feature_names),
            'n_features_out': len(self.feature_names_out)
        }
    
    
    def save(self, filepath: Path) -> None:
        """
        Save preprocessor to disk.
        
        Args:
            filepath: Path where to save the preprocessor
        """
        save_model(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    
    @staticmethod
    def load(filepath: Path) -> 'Preprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to saved preprocessor
        
        Returns:
            Loaded preprocessor
        """
        preprocessor = load_model(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    
    def summary(self) -> None:
        """Print summary of preprocessing steps."""
        print_section_header("PREPROCESSOR SUMMARY")
    
        print(f"Status: {' Fitted' if self.is_fitted else ' Not fitted'}")
        
        if self.is_fitted:
            print(f"\n📊 Features:")
            print(f"   Input features:    {len(self.original_feature_names)}")
            print(f"   Output features:   {len(self.feature_names_out)}")
            
            print(f"\n🔧 Transformations:")
            print(f"   Missing values:    {config.MISSING_STRATEGY_NUMERICAL} (numerical), "
                  f"{config.MISSING_STRATEGY_CATEGORICAL} (categorical)")
            print(f"   Encoding:          {config.ENCODING_STRATEGY}")
            print(f"   Scaling:           {config.SCALING_STRATEGY}")
            
            if config.ENCODING_STRATEGY == "onehot":
                print(f"\n Encoded Features:")
                for col, categories in self.encoding_mappings.items():
                    print(f"   {col}: {len(categories)} categories")
        
        print()


class OutlierHandler:
    """
    Optional class for handling outliers.
    Can be used before or after preprocessing.
    """
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5):
        """
        Initialize OutlierHandler.
        
        Args:
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.bounds = {}
        self.is_fitted = False
    
    
    def fit(self, X: pd.DataFrame, columns: list = None) -> 'OutlierHandler':
        """
        Fit outlier bounds on data.
        
        Args:
            X: Data to fit on
            columns: Columns to check for outliers (all numerical if None)
        
        Returns:
            Self
        """
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in X.columns:
                continue
            
            data = X[col].dropna()
            
            if self.method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
            
            elif self.method == 'zscore':
                mean = data.mean()
                std = data.std()
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            self.bounds[col] = {'lower': lower, 'upper': upper}
        
        self.is_fitted = True
        return self
    
    
    def transform(self, X: pd.DataFrame, strategy: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers in data.
        
        Args:
            X: Data to transform
            strategy: How to handle outliers ('clip', 'remove', or 'flag')
        
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise RuntimeError("OutlierHandler must be fitted first")
        
        X_transformed = X.copy()
        
        for col, bounds in self.bounds.items():
            if col not in X_transformed.columns:
                continue
            
            if strategy == 'clip':
                # Clip values to bounds
                X_transformed[col] = X_transformed[col].clip(
                    lower=bounds['lower'],
                    upper=bounds['upper']
                )
            
            elif strategy == 'remove':
                # This should be done at dataset level, not here
                logger.warning("'remove' strategy should be applied before train/test split")
            
            elif strategy == 'flag':
                # Add a binary column indicating outliers
                mask = (X_transformed[col] < bounds['lower']) | (X_transformed[col] > bounds['upper'])
                X_transformed[f'{col}_is_outlier'] = mask.astype(int)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        return X_transformed
    
    
    def fit_transform(self, X: pd.DataFrame, columns: list = None, strategy: str = 'clip') -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, columns)
        return self.transform(X, strategy)


# CONVENIENCE FUNCTIONS

def preprocess_data(X_train: pd.DataFrame, 
                   X_test: pd.DataFrame,
                   handle_outliers: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Preprocessor]:
    """
    Convenience function to preprocess train and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        handle_outliers: Whether to handle outliers
    
    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessor)
    
    Example:
        >>> X_train_clean, X_test_clean, preprocessor = preprocess_data(X_train, X_test)
    """
    # Initialize preprocessor
    preprocessor = Preprocessor()
    
    # Handle outliers if requested
    if handle_outliers and config.HANDLE_OUTLIERS:
        print("🔧 Handling outliers...")
        outlier_handler = OutlierHandler(
            method=config.OUTLIER_METHOD,
            threshold=config.OUTLIER_THRESHOLD
        )
        X_train = outlier_handler.fit_transform(X_train, strategy='clip')
        X_test = outlier_handler.transform(X_test, strategy='clip')
        print("   ✓ Outliers handled\n")
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Print summary
    preprocessor.summary()
    
    logger.info(f"Data preprocessed: Train {X_train_processed.shape}, Test {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, preprocessor


# MAIN EXECUTION (for testing)

if __name__ == "__main__":
    """Test the Preprocessor class."""
    
    
    print("TESTING PREPROCESSOR MODULE")
 
    
    # Load data
    from data_loader import get_train_test_data
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    print(f" Original shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    
    # Preprocess
    X_train_clean, X_test_clean, preprocessor = preprocess_data(
        X_train, X_test, 
        handle_outliers=True
    )
    
    print(f" Processed shapes:")
    print(f"   X_train: {X_train_clean.shape}")
    print(f"   X_test:  {X_test_clean.shape}")
    
    print(f"  Feature names after encoding:")
    feature_names = preprocessor.get_feature_names()
    print(f"   Total features: {len(feature_names)}")
    print(f"   First 10: {feature_names[:10]}")
    
    # Check for missing values
    print(f" Missing values after preprocessing:")
    print(f"   Train: {X_train_clean.isnull().sum().sum()}")
    print(f"   Test:  {X_test_clean.isnull().sum().sum()}")
    
    print("PREPROCESSOR TEST COMPLETED")
