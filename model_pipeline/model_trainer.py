"""
Model Trainer Module
Handles model training, evaluation, hyperparameter tuning, and comparison.
Supports multiple models: LightGBM, XGBoost, Random Forest, Linear Regression.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
import lightgbm as lgb
import xgboost as xgb

from . import config
from .utils import (
    setup_logger, 
    save_model, 
    load_model,
    save_metadata,
    calculate_metrics,
    print_metrics,
    compare_models,
    print_model_comparison,
    get_feature_importance,
    plot_feature_importance,
    analyze_errors,
    print_error_analysis,
    print_section_header,
    create_timestamp
)


# Setup logger
logger = setup_logger(__name__, config.TRAINING_LOG_PATH)


class ModelTrainer:
    """
    Model Trainer for food delivery time prediction.
    Handles training, evaluation, tuning, and model selection.
    
    Attributes:
        model_type: Type of model to train
        model: Trained model
        best_params: Best hyperparameters (if tuning enabled)
        cv_scores: Cross-validation scores
        train_metrics: Metrics on training set
        test_metrics: Metrics on test set
        feature_importance: Feature importance dataframe
        
    Example:
        >>> trainer = ModelTrainer(model_type="lightgbm")
        >>> trainer.train(X_train, y_train)
        >>> metrics = trainer.evaluate(X_test, y_test)
    """
    
    def __init__(self, model_type: str = None):
        """
        Initialize ModelTrainer.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'random_forest', 'linear_regression')
                       Uses config.MODEL_TYPE if None
        """
        self.model_type = model_type or config.MODEL_TYPE
        
        # Ensure model_type is string (not list)
        if isinstance(self.model_type, list):
            raise ValueError("ModelTrainer requires a single model_type. Use train_multiple_models() for comparison.")
        
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.train_metrics = None
        self.test_metrics = None
        self.feature_importance = None
        self.training_time = None
        
        logger.info(f"ModelTrainer initialized for: {self.model_type}")
    
    
    def _create_model(self, params: Dict = None) -> Any:
        """
        Create model instance based on model_type.
        
        Args:
            params: Model hyperparameters (uses config if None)
        
        Returns:
            Model instance
        """
        if params is None:
            params = config.MODEL_PARAMS.get(self.model_type, {})
        
        if self.model_type == "lightgbm":
            model = lgb.LGBMRegressor(**params)
        
        elif self.model_type == "xgboost":
            model = xgb.XGBRegressor(**params)
        
        elif self.model_type == "random_forest":
            model = RandomForestRegressor(**params)
        
        elif self.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(**params)
        
        elif self.model_type == "linear_regression":
            model = LinearRegression(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             perform_cv: bool = True,
             verbose: bool = True) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            perform_cv: Whether to perform cross-validation
            verbose: Print training information
        
        Returns:
            Trained model
        """
        print_section_header(f"TRAINING {self.model_type.upper()} MODEL")
        
        start_time = datetime.now()
        
        # Perform hyperparameter tuning if enabled
        if config.PERFORM_TUNING:
            logger.info("Hyperparameter tuning enabled")
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            # Create model with default params
            self.model = self._create_model()
        
        # Cross-validation before final training
        if perform_cv:
            self._perform_cross_validation(X_train, y_train)
        
        # Train final model
        if verbose:
            print(" Training final model...")
        
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.train_metrics = calculate_metrics(y_train, y_train_pred)
        
        if verbose:
            print(f" Training completed in {self.training_time:.2f} seconds")
            print_metrics(self.train_metrics, "TRAINING SET")
        
        logger.info(f"Model trained in {self.training_time:.2f}s")
        logger.info(f"Training RMSE: {self.train_metrics['rmse']:.4f}")
        
        return self.model
    
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform cross-validation and store scores."""
        print(" Performing cross-validation...")
        
        model = self._create_model()
        
        # Perform CV
        cv_scores = cross_val_score(
            model, X, y,
            cv=config.CV_FOLDS,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        # Convert to positive RMSE
        self.cv_scores = -cv_scores
        
        print(f"   CV RMSE: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        logger.info(f"CV scores: {self.cv_scores}")
    
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Training features
            y: Training target
        
        Returns:
            Best model
        """
        print_section_header(f"HYPERPARAMETER TUNING - {self.model_type.upper()}")
        
        base_model = self._create_model()
        param_grid = config.PARAM_GRID.get(self.model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {self.model_type}")
            return base_model
        
        if config.TUNING_METHOD == "random_search":
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=config.N_ITER_RANDOM_SEARCH,
                cv=config.CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
                verbose=1
            )
        
        elif config.TUNING_METHOD == "grid_search":
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=config.CV_FOLDS,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        
        else:
            raise ValueError(f"Unknown tuning method: {config.TUNING_METHOD}")
        
        print(f"🔍 Searching with {config.TUNING_METHOD}...")
        search.fit(X, y)
        
        self.best_params = search.best_params_
        best_score = -search.best_score_
        
        print(f"\n Best RMSE from tuning: {best_score:.4f}")
        print(f" Best parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        logger.info(f"Tuning completed. Best RMSE: {best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return search.best_estimator_
    
    
    def evaluate(self, 
                X_test: pd.DataFrame, 
                y_test: pd.Series,
                verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test target
            verbose: Print evaluation results
        
        Returns:
            Dictionary with test metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        print_section_header(f"EVALUATING {self.model_type.upper()} MODEL")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.test_metrics = calculate_metrics(y_test, y_pred)
        
        if verbose:
            print_metrics(self.test_metrics, "TEST SET")
            
            # Compare with expected performance
            if self.model_type in config.EXPECTED_PERFORMANCE:
                expected = config.EXPECTED_PERFORMANCE[self.model_type]
                print(f"📊 Comparison with research benchmarks:")
                print(f"   Expected R²:  {expected['r2']:.2f}")
                print(f"   Actual R²:    {self.test_metrics['r2']:.2f}")
                print(f"   Expected RMSE: {expected['rmse']:.2f}")
                print(f"   Actual RMSE:   {self.test_metrics['rmse']:.2f}")
                
                # Check if we're meeting expectations
                if self.test_metrics['r2'] >= expected['r2'] * 0.95:  # Within 5%
                    print(f"    Performance meets expectations!")
                else:
                    print(f"     Performance below expectations")
                print()
        
        logger.info(f"Test RMSE: {self.test_metrics['rmse']:.4f}, R²: {self.test_metrics['r2']:.4f}")
        
        return self.test_metrics
    
    
    def calculate_feature_importance(self, 
                                    feature_names: List[str],
                                    plot: bool = True,
                                    top_n: int = None) -> pd.DataFrame:
        """
        Calculate and optionally plot feature importance.
        
        Args:
            feature_names: List of feature names
            plot: Whether to plot importance
            top_n: Number of top features to show (uses config if None)
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first")
        
        print_section_header("FEATURE IMPORTANCE ANALYSIS")
        
        # Get feature importance
        self.feature_importance = get_feature_importance(self.model, feature_names)
        
        # Print top features
        top_n = top_n or config.FEATURE_IMPORTANCE_TOP_N
        print(f" Top {top_n} Most Important Features:")
        print()
        for idx, row in self.feature_importance.head(top_n).iterrows():
            print(f"   {idx+1:2d}. {row['feature']:40s} {row['importance']:10.6f}")
        print()
        
        # Plot if requested
        if plot:
            plot_feature_importance(self.feature_importance, top_n=top_n)
        
        logger.info(f"Feature importance calculated for {len(feature_names)} features")
        
        return self.feature_importance
    
    
    def analyze_prediction_errors(self,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 verbose: bool = True) -> Dict:
        """
        Analyze prediction errors.
        
        Args:
            X_test: Test features
            y_test: Test target
            verbose: Print analysis
        
        Returns:
            Dictionary with error analysis
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Analyze errors
        error_analysis = analyze_errors(
            y_test, 
            y_pred, 
            X_test,
            percentile=config.ERROR_PERCENTILE_THRESHOLD
        )
        
        if verbose:
            print_error_analysis(error_analysis)
        
        logger.info(f"Error analysis: {error_analysis['pct_large_errors']:.2f}% large errors")
        
        return error_analysis
    
    
    def save(self, 
            model_path: Path = None,
            save_metadata: bool = True) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save model (uses config if None)
            save_metadata: Whether to save metadata
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")
        
        # Use default path if not provided
        if model_path is None:
            timestamp = create_timestamp()
            model_filename = f"{self.model_type}_{timestamp}.pkl"
            model_path = config.MODEL_DIR / model_filename
        
        # Save model
        save_model(self.model, model_path)
        
        # Save metadata if requested
        if save_metadata:
            metadata = {
                'model_type': self.model_type,
                'timestamp': create_timestamp(),
                'train_metrics': self.train_metrics,
                'test_metrics': self.test_metrics,
                'cv_scores': self.cv_scores.tolist() if self.cv_scores is not None else None,
                'best_params': self.best_params,
                'training_time_seconds': self.training_time,
                'config': {
                    'random_state': config.RANDOM_STATE,
                    'test_size': config.TEST_SIZE,
                    'features': config.ALL_FEATURES
                }
            }
            
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            from utils import save_metadata as save_meta
            save_meta(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
    
    
    @staticmethod
    def load(model_path: Path) -> Tuple[Any, Dict]:
        """
        Load model and metadata from disk.
        
        Args:
            model_path: Path to saved model
        
        Returns:
            Tuple of (model, metadata)
        """
        model = load_model(model_path)
        
        # Try to load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        metadata = None
        if metadata_path.exists():
            from utils import load_metadata as load_meta
            metadata = load_meta(metadata_path)
        
        logger.info(f"Model loaded from {model_path}")
        
        return model, metadata


def train_multiple_models(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_test: pd.DataFrame,
                         y_test: pd.Series,
                         model_types: List[str] = None,
                         feature_names: List[str] = None,
                         save_models: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and compare multiple models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_types: List of model types to train (uses config if None)
        feature_names: List of feature names for importance analysis
        save_models: Whether to save trained models
    
    Returns:
        Tuple of (trained_models_dict, comparison_dataframe)
    
    Example:
        >>> models, comparison = train_multiple_models(X_train, y_train, X_test, y_test)
        >>> print(comparison)
    """
    print_section_header("TRAINING AND COMPARING MULTIPLE MODELS")
    
    model_types = model_types or config.MODELS_TO_TRAIN
    
    trained_models = {}
    results = {}
    training_times = {}
    
    # Train each model
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f" TRAINING: {model_type.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Initialize trainer
            trainer = ModelTrainer(model_type=model_type)
            
            # Train
            trainer.train(X_train, y_train, perform_cv=True, verbose=True)
            
            # Evaluate
            test_metrics = trainer.evaluate(X_test, y_test, verbose=True)
            
            # Feature importance (if available)
            if feature_names and hasattr(trainer.model, 'feature_importances_'):
                trainer.calculate_feature_importance(feature_names, plot=False)
            
            # Error analysis
            trainer.analyze_prediction_errors(X_test, y_test, verbose=True)
            
            # Store results
            trained_models[model_type] = trainer.model
            results[model_type] = test_metrics
            training_times[model_type] = trainer.training_time
            
            # Save model if requested
            if save_models:
                trainer.save()
            
            logger.info(f"{model_type} training completed successfully")
        
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            print(f" Error training {model_type}: {e}")
            continue
    
    # Compare models
    print_section_header("FINAL MODEL COMPARISON")
    
    comparison_df = compare_models(results, sort_by='rmse', ascending=True)
    print_model_comparison(comparison_df)
    
    # Add training times to comparison
    comparison_df['training_time_sec'] = [training_times.get(idx, None) for idx in comparison_df.index]
    
    # Save comparison to file
    comparison_path = config.REPORTS_DIR / f"model_comparison_{create_timestamp()}.csv"
    comparison_df.to_csv(comparison_path)
    print(f"📊 Comparison saved to: {comparison_path}\n")
    
    logger.info(f"Model comparison completed. Best model: {comparison_df.index[0]}")
    
    return trained_models, comparison_df


def get_best_model(comparison_df: pd.DataFrame,
                  trained_models: Dict,
                  metric: str = 'rmse') -> Tuple[str, Any]:
    """
    Get the best model based on a metric.
    
    Args:
        comparison_df: Comparison dataframe from train_multiple_models()
        trained_models: Dictionary of trained models
        metric: Metric to use for selection (default: 'rmse')
    
    Returns:
        Tuple of (best_model_name, best_model)
    
    Example:
        >>> best_name, best_model = get_best_model(comparison_df, models)
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison")
    
    # Sort by metric (assuming lower is better for rmse, mae, mape)
    if metric in ['rmse', 'mae', 'mape']:
        best_model_name = comparison_df[metric].idxmin()
    else:  # For r2, higher is better
        best_model_name = comparison_df[metric].idxmax()
    
    best_model = trained_models[best_model_name]
    
    print(f"🏆 Best model selected: {best_model_name.upper()}")
    print(f"   {metric.upper()}: {comparison_df.loc[best_model_name, metric]:.4f}\n")
    
    return best_model_name, best_model


# MAIN EXECUTION 

if __name__ == "__main__":
    """Test the ModelTrainer class."""
    
    print("="*80)
    print("TESTING MODEL TRAINER MODULE")
    print("="*80)
    
    # Load and prepare data
    from data_loader import get_train_test_data
    from preprocessor import preprocess_data
    from feature_engineer import engineer_features
    
    # Get data
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # Preprocess
    X_train_prep, X_test_prep, preprocessor = preprocess_data(X_train, X_test)
    
    # Engineer features
    X_train_final, X_test_final, engineer = engineer_features(X_train_prep, X_test_prep)
    
    # Get feature names
    feature_names = engineer.get_feature_names()
    
    # Train multiple models
    models, comparison = train_multiple_models(
        X_train_final, y_train,
        X_test_final, y_test,
        feature_names=feature_names,
        save_models=False
    )
    
    # Get best model
    best_name, best_model = get_best_model(comparison, models)
    

    print("MODEL TRAINER TEST COMPLETED")
