"""
Utility functions for the delivery time prediction pipeline.
Provides reusable functions for logging, model persistence, metrics, validation, and analysis.
"""

import logging
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import config (will be available after config.py is in the same directory)
try:
    import config
except ImportError:
    print("Warning: config.py not found. Some functions may not work correctly.")
    config = None


# LOGGING SETUP

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = setup_logger("model_training", "logs/training.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


# MODEL PERSISTENCE
def save_model(model: Any, filepath: Path, compressed: bool = True) -> None:
    """
    Save model to disk using joblib with optional compression.
    
    Args:
        model: Model object to save
        filepath: Path where to save the model
        compressed: Whether to compress the model (default: True)
    
    Example:
        >>> save_model(trained_model, Path("models/model_v1.pkl"))
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if compressed:
        joblib.dump(model, filepath, compress=3)
    else:
        joblib.dump(model, filepath)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath: Path) -> Any:
    """
    Load model from disk.
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded model object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    
    Example:
        >>> model = load_model(Path("models/model_v1.pkl"))
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found at: {filepath}")
    
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def save_metadata(metadata: Dict, filepath: Path) -> None:
    """
    Save metadata as JSON.
    
    Args:
        metadata: Dictionary with metadata
        filepath: Path where to save the metadata
    
    Example:
        >>> metadata = {"version": "v1.0", "date": "2025-02-05"}
        >>> save_metadata(metadata, Path("models/metadata.json"))
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    
    print(f"Metadata saved to: {filepath}")


def load_metadata(filepath: Path) -> Dict:
    """
    Load metadata from JSON.
    
    Args:
        filepath: Path to the metadata file
    
    Returns:
        Dictionary with metadata
    
    Example:
        >>> metadata = load_metadata(Path("models/metadata.json"))
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    return metadata


# METRICS CALCULATION
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metric names and values
    
    Example:
        >>> metrics = calculate_metrics(y_test, predictions)
        >>> print(f"RMSE: {metrics['rmse']:.2f}")
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }
    
    return metrics


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value as percentage
    
    Note:
        Returns NaN if y_true contains zeros (division by zero)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def print_metrics(metrics: Dict[str, float], dataset_name: str = "", prefix: str = "") -> None:
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary with metric names and values
        dataset_name: Name of the dataset (e.g., "Train", "Test")
        prefix: Prefix to add to the output
    
    Example:
        >>> print_metrics(metrics, "Test Set")
    """
    print(f"\n{'='*70}")
    print(f"{prefix} METRICS {dataset_name}")
    print(f"{'='*70}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():10s}: {value:10.4f}")
    print(f"{'='*70}\n")


# MODEL COMPARISON
def compare_models(results: Dict[str, Dict[str, float]], 
                   sort_by: str = "rmse",
                   ascending: bool = True) -> pd.DataFrame:
    """
    Compare multiple models and rank them by performance.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
                 Example: {'lightgbm': {'rmse': 5.2, 'mae': 3.8, ...}, ...}
        sort_by: Metric to sort by (default: 'rmse')
        ascending: Sort order (True for metrics where lower is better)
    
    Returns:
        DataFrame with models ranked by performance
    
    Example:
        >>> results = {"lightgbm": {"rmse": 5.2}, "xgboost": {"rmse": 5.5}}
        >>> comparison = compare_models(results)
    """
    df = pd.DataFrame(results).T
    
    # Sort by specified metric
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    # Add rank column
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    return df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Pretty print model comparison results.
    
    Args:
        comparison_df: DataFrame from compare_models()
    
    Example:
        >>> print_model_comparison(comparison_df)
    """
    print("\n" + "="*80)
    print("🏆 MODEL COMPARISON RESULTS")
    print("="*80)
    print(comparison_df.to_string())
    print("="*80)
    
    # Highlight best model
    if len(comparison_df) > 0:
        best_model = comparison_df.index[0]
        best_rmse = comparison_df.iloc[0].get('rmse', 'N/A')
        print(f"\n BEST MODEL: {best_model.upper()} (RMSE: {best_rmse})")
        print("="*80 + "\n")


# DATA VALIDATION
def validate_input_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that input dataframe has required columns.
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If missing required columns
    
    Example:
        >>> validate_input_data(df, ["Distance_km", "Weather"])
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f" Missing required columns: {missing_cols}")
    return True


def validate_value_ranges(df: pd.DataFrame, valid_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Validate numerical values are within expected ranges.
    
    Args:
        df: Input dataframe
        valid_ranges: Dictionary mapping column names to (min, max) tuples
    
    Returns:
        DataFrame with validation report
    
    Example:
        >>> ranges = {"Distance_km": (0.1, 50.0)}
        >>> report = validate_value_ranges(df, ranges)
    """
    validation_report = []
    
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            validation_report.append({
                'column': col,
                'min_expected': min_val,
                'max_expected': max_val,
                'min_actual': df[col].min(),
                'max_actual': df[col].max(),
                'out_of_range_count': out_of_range,
                'out_of_range_pct': (out_of_range / len(df)) * 100
            })
    
    return pd.DataFrame(validation_report)


def check_data_drift(reference_data: pd.DataFrame, 
                     new_data: pd.DataFrame, 
                     threshold: float = 0.05) -> Dict[str, Dict]:
    """
    Simple data drift detection using Kolmogorov-Smirnov test.
    
    Args:
        reference_data: Reference dataset (e.g., training data)
        new_data: New dataset to check for drift
        threshold: P-value threshold for detecting drift (default: 0.05)
    
    Returns:
        Dictionary with column names and drift detection results
    
    Example:
        >>> drift_report = check_data_drift(train_df, new_df)
        >>> for col, result in drift_report.items():
        ...     if result['drift_detected']:
        ...         print(f" D.rift detected in {col}")
    """
    from scipy.stats import ks_2samp
    
    drift_results = {}
    
    for col in reference_data.columns:
        if reference_data[col].dtype in ['float64', 'int64']:
            try:
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(
                    reference_data[col].dropna(), 
                    new_data[col].dropna()
                )
                
                drift_results[col] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold,
                    'severity': 'high' if p_value < 0.01 else 'medium' if p_value < threshold else 'low'
                }
            except Exception as e:
                drift_results[col] = {
                    'error': str(e),
                    'drift_detected': None
                }
    
    return drift_results


# FEATURE IMPORTANCE
def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    
    Returns:
        DataFrame with features sorted by importance
    
    Example:
        >>> importance_df = get_feature_importance(model, X.columns.tolist())
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importance = model.feature_importance()
    else:
        print(" Model doesn't have feature importance attribute")
        return pd.DataFrame()
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, 
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame from get_feature_importance()
        top_n: Number of top features to plot
        figsize: Figure size (width, height)
    
    Example:
        >>> importance_df = get_feature_importance(model, features)
        >>> plot_feature_importance(importance_df, top_n=15)
    """
    import matplotlib.pyplot as plt
    
    # Select top N features
    plot_df = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(range(len(plot_df)), plot_df['importance'], color='steelblue')
    plt.yticks(range(len(plot_df)), plot_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ERROR ANALYSIS
def analyze_errors(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   X: pd.DataFrame = None,
                   percentile: float = 90) -> Dict:
    """
    Analyze prediction errors.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        X: Feature dataframe (optional)
        percentile: Percentile for identifying large errors
    
    Returns:
        Dictionary with error analysis
    
    Example:
        >>> error_report = analyze_errors(y_test, predictions, X_test)
        >>> print(f"Large errors: {error_report['n_large_errors']}")
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    threshold = np.percentile(abs_errors, percentile)
    large_errors_idx = abs_errors > threshold
    
    analysis = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'large_error_threshold': threshold,
        'n_large_errors': np.sum(large_errors_idx),
        'pct_large_errors': (np.sum(large_errors_idx) / len(errors)) * 100,
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors)
    }
    
    # Underestimation vs overestimation
    analysis['n_underestimations'] = np.sum(errors < 0)
    analysis['n_overestimations'] = np.sum(errors > 0)
    analysis['pct_underestimations'] = (analysis['n_underestimations'] / len(errors)) * 100
    analysis['pct_overestimations'] = (analysis['n_overestimations'] / len(errors)) * 100
    
    if X is not None:
        # Find patterns in large errors
        analysis['large_error_samples'] = X[large_errors_idx].describe().to_dict()
        analysis['large_error_indices'] = np.where(large_errors_idx)[0].tolist()
    
    return analysis


def print_error_analysis(error_analysis: Dict) -> None:
    """
    Pretty print error analysis results.
    
    Args:
        error_analysis: Dictionary from analyze_errors()
    """
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    print(f"Mean Error:             {error_analysis['mean_error']:10.4f}")
    print(f"Std Error:              {error_analysis['std_error']:10.4f}")
    print(f"Median Error:           {error_analysis['median_error']:10.4f}")
    print(f"Max Error:              {error_analysis['max_error']:10.4f}")
    print(f"\nLarge Error Threshold:  {error_analysis['large_error_threshold']:10.4f}")
    print(f"Large Errors Count:     {error_analysis['n_large_errors']:10d} ({error_analysis['pct_large_errors']:.2f}%)")
    print(f"\nUnderestimations:       {error_analysis['n_underestimations']:10d} ({error_analysis['pct_underestimations']:.2f}%)")
    print(f"Overestimations:        {error_analysis['n_overestimations']:10d} ({error_analysis['pct_overestimations']:.2f}%)")
    print("="*70 + "\n")


# UTILITY FUNCTIONS

def create_timestamp() -> str:
    """
    Create timestamp string for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    
    Example:
        >>> timestamp = create_timestamp()
        >>> print(timestamp)  # '20250205_143022'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate memory usage of dataframe.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with memory usage in MB
    
    Example:
        >>> usage = memory_usage(df)
        >>> print(f"Total: {usage['total_mb']:.2f} MB")
    """
    mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    return {
        'total_mb': mem_usage,
        'per_column': (df.memory_usage(deep=True) / 1024**2).to_dict()
    }


def print_section_header(title: str, char: str = "=") -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character to use for the line
    
    Example:
        >>> print_section_header("MODEL TRAINING")
    """
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")