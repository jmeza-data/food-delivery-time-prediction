"""
Predictor Module
Handles predictions in production environment.
Loads saved models and preprocessing pipelines to make predictions on new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from . import config
from .utils import (
    setup_logger,
    load_model,
    load_metadata,
    validate_input_data,
    validate_value_ranges,
    print_section_header
)


# Setup logger
logger = setup_logger(__name__, config.PREDICTION_LOG_PATH if config.LOG_PREDICTIONS else None)


class Predictor:
    """
    Production-ready predictor for food delivery time.
    Loads trained model and preprocessing pipeline to make predictions.
    
    Attributes:
        model: Trained prediction model
        preprocessor: Fitted preprocessor
        feature_engineer: Fitted feature engineer
        model_metadata: Model metadata
        is_loaded: Whether components are loaded
        
    Example:
        >>> predictor = Predictor()
        >>> predictor.load_pipeline()
        >>> delivery_time = predictor.predict(new_order)
    """
    
    def __init__(self,
                 model_path: Path = None,
                 preprocessor_path: Path = None,
                 feature_engineer_path: Path = None):
        """
        Initialize Predictor.
        
        Args:
            model_path: Path to saved model (uses config if None)
            preprocessor_path: Path to saved preprocessor (uses config if None)
            feature_engineer_path: Path to saved feature engineer (uses config if None)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.feature_engineer_path = feature_engineer_path
        
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model_metadata = None
        
        self.is_loaded = False
        
        logger.info("Predictor initialized")
    
    
    def load_pipeline(self,
                     model_path: Path = None,
                     preprocessor_path: Path = None,
                     feature_engineer_path: Path = None,
                     verbose: bool = True) -> None:
        """
        Load complete prediction pipeline (model + preprocessor + feature engineer).
        
        Args:
            model_path: Path to model file
            preprocessor_path: Path to preprocessor file
            feature_engineer_path: Path to feature engineer file
            verbose: Print loading information
        """
        print_section_header("LOADING PREDICTION PIPELINE")
        
        # Use provided paths or stored paths
        model_path = model_path or self.model_path
        preprocessor_path = preprocessor_path or self.preprocessor_path
        feature_engineer_path = feature_engineer_path or self.feature_engineer_path
        
        # Load model
        if model_path is None:
            # Try to find the latest model
            model_path = self._find_latest_model()
        
        if model_path and model_path.exists():
            self.model = load_model(model_path)
            self.model_path = model_path
            
            # Try to load metadata
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                self.model_metadata = load_metadata(metadata_path)
                if verbose:
                    print(f"📋 Model: {self.model_metadata.get('model_type', 'unknown')}")
                    print(f"   Trained: {self.model_metadata.get('timestamp', 'unknown')}")
                    if 'test_metrics' in self.model_metadata:
                        metrics = self.model_metadata['test_metrics']
                        print(f"   Test RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                        print(f"   Test R²: {metrics.get('r2', 'N/A'):.4f}")
        else:
            raise FileNotFoundError(f" Model not found at: {model_path}")
        
        # Load preprocessor
        if preprocessor_path and preprocessor_path.exists():
            from .preprocessor import Preprocessor
            self.preprocessor = Preprocessor.load(preprocessor_path)
            self.preprocessor_path = preprocessor_path
        elif verbose:
            print("  Preprocessor not provided - raw data must be preprocessed")
        
        # Load feature engineer
        if feature_engineer_path and feature_engineer_path.exists():
            from .feature_engineer import FeatureEngineer
            self.feature_engineer = FeatureEngineer.load(feature_engineer_path)
            self.feature_engineer_path = feature_engineer_path
        elif verbose:
            print("⚠️  Feature engineer not provided - features must be pre-engineered")
        
        self.is_loaded = True
        
        if verbose:
            print("\n Pipeline loaded successfully\n")
        
        logger.info("Prediction pipeline loaded successfully")
    
    
    def _find_latest_model(self) -> Optional[Path]:
        """
        Find the most recent model file in the models directory.
        
        Returns:
            Path to latest model or None
        """
        model_dir = config.MODEL_DIR
        if not model_dir.exists():
            return None
        
        # Find all .pkl files
        model_files = list(model_dir.glob("*.pkl"))
        
        # Exclude preprocessor and feature_engineer files
        model_files = [f for f in model_files 
                      if 'preprocessor' not in f.name.lower() 
                      and 'feature_engineer' not in f.name.lower()]
        
        if not model_files:
            return None
        
        # Return the most recent
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return latest
    
    
    def predict(self,
               X: Union[pd.DataFrame, Dict, List[Dict]],
               return_dict: bool = False) -> Union[np.ndarray, List[Dict]]:
        """
        Make predictions on new data.
        
        Args:
            X: Input data (DataFrame, single dict, or list of dicts)
            return_dict: Whether to return results as dictionary with details
        
        Returns:
            Predictions (array or list of dicts)
        
        Example:
            >>> # Single prediction
            >>> order = {
            ...     'Distance_km': 5.2,
            ...     'Weather': 'Clear',
            ...     'Traffic_Level': 'Medium',
            ...     ...
            ... }
            >>> prediction = predictor.predict(order)
            
            >>> # Batch prediction
            >>> orders = [order1, order2, order3]
            >>> predictions = predictor.predict(orders)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(" Pipeline not loaded. Call load_pipeline() first.")
        
        # Convert input to DataFrame
        X_df = self._prepare_input(X)
        
        # Validate input
        self._validate_input(X_df)
        
        # Apply preprocessing pipeline
        X_processed = self._preprocess(X_df)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Log predictions if enabled
        if config.LOG_PREDICTIONS:
            self._log_predictions(X_df, predictions)
        
        # Return format
        if return_dict:
            results = []
            for idx, pred in enumerate(predictions):
                result = {
                    'predicted_delivery_time_min': float(pred),
                    'input_data': X_df.iloc[idx].to_dict() if isinstance(X_df, pd.DataFrame) else X_df
                }
                results.append(result)
            return results if len(results) > 1 else results[0]
        
        return predictions
    
    
    def predict_single(self, order: Dict) -> float:
        """
        Convenience method for single prediction.
        
        Args:
            order: Dictionary with order details
        
        Returns:
            Predicted delivery time in minutes
        
        Example:
            >>> time = predictor.predict_single({
            ...     'Distance_km': 5.2,
            ...     'Weather': 'Clear',
            ...     'Traffic_Level': 'Medium',
            ...     'Time_of_Day': 'Evening',
            ...     'Vehicle_Type': 'Scooter',
            ...     'Preparation_Time_min': 15,
            ...     'Courier_Experience_yrs': 3.5
            ... })
            >>> print(f"Estimated delivery time: {time:.1f} minutes")
        """
        prediction = self.predict(order)
        return float(prediction[0])
    
    
    def predict_with_confidence(self,
                               X: Union[pd.DataFrame, Dict],
                               n_estimators: int = None) -> Dict:
        """
        Make prediction with confidence interval (for tree-based models).
        
        Args:
            X: Input data
            n_estimators: Number of estimators to use for uncertainty (None = all)
        
        Returns:
            Dictionary with prediction, lower_bound, upper_bound
        
        Note:
            Only works with ensemble models (Random Forest, GradientBoosting, etc.)
        """
        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Confidence intervals only available for ensemble models")
        
        # Prepare input
        X_df = self._prepare_input(X)
        X_processed = self._preprocess(X_df)
        
        # Get predictions from individual estimators
        estimators = self.model.estimators_[:n_estimators] if n_estimators else self.model.estimators_
        
        predictions = []
        for estimator in estimators:
            pred = estimator.predict(X_processed)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 95% confidence interval (approximately ±2 std)
        lower_bound = mean_pred - 2 * std_pred
        upper_bound = mean_pred + 2 * std_pred
        
        result = {
            'prediction': float(mean_pred[0]),
            'lower_bound_95': float(lower_bound[0]),
            'upper_bound_95': float(upper_bound[0]),
            'std_dev': float(std_pred[0])
        }
        
        return result
    
    
    def _prepare_input(self, X: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Convert input to DataFrame format.
        
        Args:
            X: Input in various formats
        
        Returns:
            DataFrame
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        
        elif isinstance(X, dict):
            return pd.DataFrame([X])
        
        elif isinstance(X, list):
            return pd.DataFrame(X)
        
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
    
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            X: Input dataframe
        
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        try:
            validate_input_data(X, config.ALL_FEATURES)
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            raise
        
        # Validate value ranges
        if config.VALID_RANGES:
            validation_report = validate_value_ranges(X, config.VALID_RANGES)
            
            # Check for out-of-range values
            total_out_of_range = validation_report['out_of_range_count'].sum()
            if total_out_of_range > 0:
                logger.warning(f"Found {total_out_of_range} out-of-range values")
                # Could raise error or just warn depending on requirements
        
        # Validate categorical values
        if config.VALID_CATEGORIES:
            for col, valid_values in config.VALID_CATEGORIES.items():
                if col in X.columns:
                    invalid_mask = ~X[col].isin(valid_values + [np.nan, None])
                    if invalid_mask.any():
                        invalid_values = X.loc[invalid_mask, col].unique()
                        logger.warning(f"Invalid values in {col}: {invalid_values}")
    
    
    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing pipeline to input data.
        
        Args:
            X: Raw input data
        
        Returns:
            Preprocessed and engineered features
        """
        X_processed = X.copy()
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X_processed)
        
        # Apply feature engineering if available
        if self.feature_engineer is not None:
            X_processed = self.feature_engineer.transform(X_processed)
        
        return X_processed
    
    
    def _log_predictions(self, X: pd.DataFrame, predictions: np.ndarray) -> None:
        """
        Log predictions for monitoring.
        
        Args:
            X: Input data
            predictions: Model predictions
        """
        import json
        from datetime import datetime
        
        timestamp = datetime.now().isoformat()
        
        for idx, pred in enumerate(predictions):
            log_entry = {
                'timestamp': timestamp,
                'prediction': float(pred),
                'input': X.iloc[idx].to_dict() if isinstance(X, pd.DataFrame) else X
            }
            logger.info(json.dumps(log_entry))
    
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded")
        
        info = {
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None,
            'feature_engineer_loaded': self.feature_engineer is not None,
            'model_type': type(self.model).__name__ if self.model else None,
        }
        
        if self.model_metadata:
            info['metadata'] = self.model_metadata
        
        return info
    
    
    def summary(self) -> None:
        """Print summary of loaded pipeline."""
        print_section_header("PREDICTOR SUMMARY")
        
        print(f"Status: {' Loaded' if self.is_loaded else ' Not loaded'}")
        
        if self.is_loaded:
            print(f"\n Components:")
            print(f"   Model:             {' Loaded' if self.model else ' Not loaded'}")
            print(f"   Preprocessor:      {' Loaded' if self.preprocessor else ' Not loaded'}")
            print(f"   Feature Engineer:  {' Loaded' if self.feature_engineer else ' Not loaded'}")
            
            if self.model_metadata:
                print(f"\n Model Info:")
                print(f"   Type:              {self.model_metadata.get('model_type', 'unknown')}")
                print(f"   Trained:           {self.model_metadata.get('timestamp', 'unknown')}")
                
                if 'test_metrics' in self.model_metadata:
                    metrics = self.model_metadata['test_metrics']
                    print(f"   Test RMSE:         {metrics.get('rmse', 'N/A'):.4f}")
                    print(f"   Test R²:           {metrics.get('r2', 'N/A'):.4f}")
                    print(f"   Test MAE:          {metrics.get('mae', 'N/A'):.4f}")
        
        print()


# CONVENIENCE FUNCTIONS

def quick_predict(order: Dict,
                 model_path: Path = None,
                 preprocessor_path: Path = None,
                 feature_engineer_path: Path = None) -> float:
    """
    Quick prediction without manually loading pipeline.
    
    Args:
        order: Order details dictionary
        model_path: Path to model (auto-detects if None)
        preprocessor_path: Path to preprocessor
        feature_engineer_path: Path to feature engineer
    
    Returns:
        Predicted delivery time in minutes
    
    Example:
        >>> time = quick_predict({
        ...     'Distance_km': 5.2,
        ...     'Weather': 'Clear',
        ...     'Traffic_Level': 'Medium',
        ...     'Time_of_Day': 'Evening',
        ...     'Vehicle_Type': 'Scooter',
        ...     'Preparation_Time_min': 15,
        ...     'Courier_Experience_yrs': 3.5
        ... })
    """
    predictor = Predictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        feature_engineer_path=feature_engineer_path
    )
    
    predictor.load_pipeline(verbose=False)
    
    return predictor.predict_single(order)


def batch_predict(orders: List[Dict],
                 model_path: Path = None,
                 preprocessor_path: Path = None,
                 feature_engineer_path: Path = None,
                 save_results: bool = False,
                 output_path: Path = None) -> pd.DataFrame:
    """
    Batch prediction on multiple orders.
    
    Args:
        orders: List of order dictionaries
        model_path: Path to model
        preprocessor_path: Path to preprocessor
        feature_engineer_path: Path to feature engineer
        save_results: Whether to save predictions to file
        output_path: Path to save results
    
    Returns:
        DataFrame with orders and predictions
    
    Example:
        >>> orders = [order1, order2, order3]
        >>> results = batch_predict(orders, save_results=True)
    """
    predictor = Predictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        feature_engineer_path=feature_engineer_path
    )
    
    predictor.load_pipeline(verbose=False)
    
    # Convert to DataFrame
    orders_df = pd.DataFrame(orders)
    
    # Make predictions
    predictions = predictor.predict(orders_df)
    
    # Add predictions to dataframe
    results_df = orders_df.copy()
    results_df['Predicted_Delivery_Time_min'] = predictions
    
    # Save if requested
    if save_results:
        if output_path is None:
            from .utils import create_timestamp
            output_path = config.BASE_DIR / f"predictions_{create_timestamp()}.csv"
        
        results_df.to_csv(output_path, index=False)
        print(f" Predictions saved to: {output_path}")
    
    return results_df


# MAIN EXECUTION (for testing)

if __name__ == "__main__":
    """Test the Predictor class."""
    
    print("="*80)
    print("TESTING PREDICTOR MODULE")
    print("="*80)
    
    # Create sample order
    sample_order = {
        'Distance_km': 5.2,
        'Weather': 'Clear',
        'Traffic_Level': 'Medium',
        'Time_of_Day': 'Evening',
        'Vehicle_Type': 'Scooter',
        'Preparation_Time_min': 15,
        'Courier_Experience_yrs': 3.5
    }
    
    print("\n📦 Sample Order:")
    for key, value in sample_order.items():
        print(f"   {key:30s}: {value}")
    
    # Try to make prediction
    try:
        print("\n🔮 Making prediction...")
        
        # Initialize predictor
        predictor = Predictor()
        
        # Try to load pipeline
        predictor.load_pipeline(verbose=True)
        
        # Make prediction
        predicted_time = predictor.predict_single(sample_order)
        
        print(f"\n Predicted delivery time: {predicted_time:.1f} minutes")
        
        # Show pipeline summary
        predictor.summary()
        
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("\n To use the predictor, you need to:")
        print("   1. Train a model using model_trainer.py")
        print("   2. Save the model, preprocessor, and feature_engineer")
        print("   3. Then load them with the predictor")
    
    except Exception as e:
        print(f"\n Error: {e}")
    
 
    print("PREDICTOR TEST COMPLETED")
