"""
Food Delivery Time Prediction Pipeline
======================================

A production-ready machine learning pipeline for predicting food delivery times.

Modules:
--------
- config: Configuration and hyperparameters
- utils: Utility functions for logging, metrics, persistence
- data_loader: Data loading and validation
- preprocessor: Data cleaning, encoding, scaling
- feature_engineer: Feature creation and engineering
- model_trainer: Model training, evaluation, comparison
- predictor: Production inference

Quick Start:
-----------
>>> # Complete pipeline
>>> from model_pipeline.data_loader import get_train_test_data
>>> from model_pipeline.preprocessor import preprocess_data
>>> from model_pipeline.feature_engineer import engineer_features
>>> from model_pipeline.model_trainer import train_multiple_models
>>> 
>>> # 1. Load data
>>> X_train, X_test, y_train, y_test = get_train_test_data()
>>> 
>>> # 2. Preprocess
>>> X_train_prep, X_test_prep, preprocessor = preprocess_data(X_train, X_test)
>>> 
>>> # 3. Engineer features
>>> X_train_final, X_test_final, engineer = engineer_features(X_train_prep, X_test_prep)
>>> 
>>> # 4. Train models
>>> models, comparison = train_multiple_models(X_train_final, y_train, X_test_final, y_test)

>>> # Make predictions
>>> from model_pipeline.predictor import Predictor
>>> predictor = Predictor()
>>> predictor.load_pipeline()
>>> time = predictor.predict_single(order_dict)

Authors: [Your Name]
Version: 1.0
Date: 2025-02-05
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main classes for convenient access
from .data_loader import DataLoader, load_delivery_data, get_train_test_data
from .preprocessor import Preprocessor, preprocess_data, OutlierHandler
from .feature_engineer import FeatureEngineer, engineer_features
from .model_trainer import ModelTrainer, train_multiple_models, get_best_model
from .predictor import Predictor, quick_predict, batch_predict

# Import config for easy access
from . import config

# Import key utilities
from .utils import (
    setup_logger,
    calculate_metrics,
    print_metrics,
    save_model,
    load_model
)

# Define what gets imported with "from model_pipeline import *"
__all__ = [
    # Main classes
    'DataLoader',
    'Preprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'Predictor',
    
    # Convenience functions - Data
    'load_delivery_data',
    'get_train_test_data',
    
    # Convenience functions - Preprocessing
    'preprocess_data',
    'OutlierHandler',
    
    # Convenience functions - Feature Engineering
    'engineer_features',
    
    # Convenience functions - Training
    'train_multiple_models',
    'get_best_model',
    
    # Convenience functions - Prediction
    'quick_predict',
    'batch_predict',
    
    # Config
    'config',
    
    # Key utilities
    'setup_logger',
    'calculate_metrics',
    'print_metrics',
    'save_model',
    'load_model',
]


def get_version():
    """Get the version of the pipeline."""
    return __version__


def print_pipeline_info():
    """Print information about the pipeline."""
    print("="*80)
    print("FOOD DELIVERY TIME PREDICTION PIPELINE")
    print("="*80)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    print("Available Modules:")
    print("  - data_loader: Load and validate delivery data")
    print("  - preprocessor: Clean, encode, and scale features")
    print("  - feature_engineer: Create interaction and domain features")
    print("  - model_trainer: Train and compare ML models")
    print("  - predictor: Make predictions on new orders")
    print()
    print("Supported Models:")
    print("  - LightGBM (recommended)")
    print("  - XGBoost")
    print("  - Random Forest")
    print("  - Linear Regression (baseline)")
    print()
    print("Quick Start:")
    print("  >>> from model_pipeline import get_train_test_data, preprocess_data")
    print("  >>> X_train, X_test, y_train, y_test = get_train_test_data()")
    print("  >>> X_train_clean, X_test_clean, pp = preprocess_data(X_train, X_test)")
    print("="*80)


# Optional: Print info when package is imported
# Uncomment the line below if you want info printed on import
# print_pipeline_info()