"""
Configuration file for the delivery time prediction pipeline.
Centralizes all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# PATHS

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# DATA CONFIGURATION
DATA_PATH = DATA_DIR / "Food_Delivery_Times.csv"
TARGET_COLUMN = "Delivery_Time_min"

NUMERICAL_FEATURES = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

CATEGORICAL_FEATURES = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# PREPROCESSING CONFIGURATION


# Missing value strategies
MISSING_STRATEGY_NUMERICAL = "median" 
MISSING_STRATEGY_CATEGORICAL = "most_frequent"

# Encoding strategy
ENCODING_STRATEGY = "onehot"  

# Scaling strategy
SCALING_STRATEGY = "standard"  

# Outlier handling
HANDLE_OUTLIERS = True
OUTLIER_METHOD = "iqr"
OUTLIER_THRESHOLD = 1.5  


# FEATURE ENGINEERING
CREATE_INTERACTIONS = True
INTERACTION_FEATURES = [
    ("Distance_km", "Traffic_Level"),
    ("Weather", "Traffic_Level"),
    ("Distance_km", "Vehicle_Type"),
]


# MODEL CONFIGURATION

# Train/test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15  # From training set
RANDOM_STATE = 42

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Train/test split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15  # From training set
RANDOM_STATE = 42

# Model selection
# Option 1: Single model (fast iteration)
# MODEL_TYPE = "lightgbm"

# Option 2: Multiple models (comparison) - RECOMMENDED FOR DELIVERABLE
MODEL_TYPE = ["lightgbm", "xgboost", "random_forest", "linear_regression"]

# Model selection logic
TRAIN_MULTIPLE_MODELS = isinstance(MODEL_TYPE, list)
MODELS_TO_TRAIN = MODEL_TYPE if TRAIN_MULTIPLE_MODELS else [MODEL_TYPE]

# Expected performance ranges (based on academic research)
EXPECTED_PERFORMANCE = {
    "lightgbm": {"r2": 0.76, "rmse": 9.5, "source": "Research paper 2025"},
    "xgboost": {"r2": 0.74, "rmse": 10.0, "source": "Research paper 2024"},
    "random_forest": {"r2": 0.70, "rmse": 11.0, "source": "Multiple studies"},
    "linear_regression": {"r2": 0.55, "rmse": 13.5, "source": "Baseline"}
}

# Model hyperparameters (optimized based on research)
MODEL_PARAMS = {
    "lightgbm": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 40,              # Optimized from research (31-50 range)
        "learning_rate": 0.08,         # Optimized (0.05-0.1 range)
        "feature_fraction": 0.8,       # Feature sampling
        "bagging_fraction": 0.8,       # Data sampling
        "bagging_freq": 5,
        "max_depth": -1,               # No limit (typical for LightGBM)
        "min_child_samples": 20,
        "n_estimators": 300,           # Increased based on research
        "verbose": -1,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "objective": "reg:squarederror",
        "max_depth": 7,                # Optimized from research (6-8 range)
        "learning_rate": 0.08,         # Optimized
        "n_estimators": 300,           # Increased based on research
        "subsample": 0.8,              # Row sampling
        "colsample_bytree": 0.8,       # Column sampling
        "min_child_weight": 3,         # Regularization
        "gamma": 0.1,                  # Minimum loss reduction
        "reg_alpha": 0.0,              # L1 regularization
        "reg_lambda": 1.0,             # L2 regularization
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 300,           # Increased from 200
        "max_depth": 20,               # Optimized from research
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",        # sqrt of n_features
        "bootstrap": True,
        "oob_score": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1                   # Use all CPU cores
    },
    "gradient_boosting": {
        "loss": "squared_error",
        "learning_rate": 0.08,
        "n_estimators": 300,
        "max_depth": 7,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "subsample": 0.8,
        "random_state": RANDOM_STATE
    },
    "linear_regression": {
        # Baseline model - no hyperparameters
        "fit_intercept": True,
        "normalize": False
    }
}

# HYPERPARAMETER TUNING
PERFORM_TUNING = False  
TUNING_METHOD = "random_search"  
CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 30

# Search space for hyperparameter tuning (if PERFORM_TUNING = True)
PARAM_GRID = {
    "lightgbm": {
        "num_leaves": [20, 31, 40, 50, 70],
        "learning_rate": [0.01, 0.05, 0.08, 0.1],
        "max_depth": [-1, 10, 20, 30],
        "min_child_samples": [10, 20, 30],
        "feature_fraction": [0.6, 0.8, 1.0],
        "bagging_fraction": [0.6, 0.8, 1.0],
        "n_estimators": [100, 200, 300, 500]
    },
    "xgboost": {
        "max_depth": [5, 6, 7, 8, 10],
        "learning_rate": [0.01, 0.05, 0.08, 0.1],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2]
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
}

# EVALUATION METRICS
PRIMARY_METRIC = "rmse"  
METRICS = ["rmse", "mae", "r2", "mape"]

# Business thresholds
ACCEPTABLE_ERROR_MINUTES = 10  
CRITICAL_ERROR_MINUTES = 20    
# Metric display names
METRIC_NAMES = {
    "rmse": "Root Mean Squared Error",
    "mae": "Mean Absolute Error",
    "r2": "R² Score",
    "mape": "Mean Absolute Percentage Error"
}

# EXPLAINABILITY
CALCULATE_SHAP = True
SHAP_SAMPLE_SIZE = 500  # Number of samples for SHAP calculation (computational cost)
SHAP_PLOT_TOP_N = 20    # Top N features to plot

# Feature importance calculation
CALCULATE_FEATURE_IMPORTANCE = True
FEATURE_IMPORTANCE_TOP_N = 15

# PRODUCTION SETTINGS
MODEL_VERSION = "v1.0"
MODEL_NAME = f"delivery_time_model_{MODEL_VERSION}.pkl"
PREPROCESSOR_NAME = f"preprocessor_{MODEL_VERSION}.pkl"
FEATURE_ENGINEER_NAME = f"feature_engineer_{MODEL_VERSION}.pkl"

# Model metadata
MODEL_METADATA = {
    "version": MODEL_VERSION,
    "created_date": None,  # Will be filled during training
    "model_type": None,    # Will be filled during training
    "features_used": ALL_FEATURES,
    "target": TARGET_COLUMN,
    "random_state": RANDOM_STATE
}

# MONITORING & LOGGING
LOG_PREDICTIONS = True
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PREDICTION_LOG_PATH = LOG_DIR / "predictions.log"
TRAINING_LOG_PATH = LOG_DIR / "training.log"

# Logging configuration
LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ERROR ANALYSIS
ERROR_ANALYSIS_ENABLED = True
ERROR_PERCENTILE_THRESHOLD = 90  # Analyze top 10% worst predictions

# DATA VALIDATION
# Value ranges for input validation (production safety)
VALID_RANGES = {
    "Distance_km": (0.1, 50.0),
    "Preparation_Time_min": (1, 120),
    "Courier_Experience_yrs": (0, 20)
}

VALID_CATEGORIES = {
    "Weather": ["Clear", "Cloudy", "Rainy", "Snowy", "Foggy", "Windy"],
    "Traffic_Level": ["Low", "Medium", "High"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
    "Vehicle_Type": ["Bike", "Scooter", "Car"]
}

# REPRODUCIBILITY
# Set seeds for reproducibility
import numpy as np
import random

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
