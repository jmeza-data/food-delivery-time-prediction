"""
Main script to run the complete delivery time prediction pipeline.
"""

from pathlib import Path
import pandas as pd

from model_pipeline.data_loader import get_train_test_data
from model_pipeline.preprocessor import preprocess_data
from model_pipeline.feature_engineer import engineer_features
from model_pipeline.model_trainer import train_multiple_models, get_best_model
from model_pipeline.predictor import Predictor
from model_pipeline import config


def main():
    """Run the complete pipeline."""
    
    print("="*80)
    print("FOOD DELIVERY TIME PREDICTION PIPELINE")
    print("="*80)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    X_train_prep, X_test_prep, preprocessor = preprocess_data(
        X_train, X_test,
        handle_outliers=config.HANDLE_OUTLIERS
    )
    
    # Save preprocessor
    preprocessor_path = config.MODEL_DIR / config.PREPROCESSOR_NAME
    preprocessor.save(preprocessor_path)
    
    # 3. Feature Engineering
    print("\n[3/5] Feature engineering...")
    X_train_final, X_test_final, engineer = engineer_features(
        X_train_prep, X_test_prep
    )
    
    # Save feature engineer
    engineer_path = config.MODEL_DIR / config.FEATURE_ENGINEER_NAME
    engineer.save(engineer_path)
    
    # 4. Train models
    print("\n[4/5] Training models...")
    models, comparison = train_multiple_models(
        X_train_final, y_train,
        X_test_final, y_test,
        feature_names=engineer.get_feature_names(),
        save_models=True
    )
    
    # 5. Select best model
    print("\n[5/5] Selecting best model...")
    best_name, best_model = get_best_model(comparison, models)
    
    # Save best model with standard name
    from model_pipeline.utils import save_model
    best_model_path = config.MODEL_DIR / config.MODEL_NAME
    save_model(best_model, best_model_path)
    
    print("\n" + "="*80)
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\n Best Model: {best_name.upper()}")
    print(f" Model saved to: {best_model_path}")
    print(f" Preprocessor saved to: {preprocessor_path}")
    print(f" Feature Engineer saved to: {engineer_path}")
    print("\n You can now use the predictor to make predictions!")
    
    return best_model, preprocessor, engineer, comparison


if __name__ == "__main__":
    main()