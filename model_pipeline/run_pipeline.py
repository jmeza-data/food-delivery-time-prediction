"""
Run complete ML pipeline for food delivery time prediction
"""

from pathlib import Path
from model_pipeline import (
    get_train_test_data,
    preprocess_data,
    engineer_features,
    train_multiple_models,
    get_best_model,
    config
)
from model_pipeline.utils import save_model

def main():
    """Run the complete pipeline."""
    
    
    print(" FOOD DELIVERY TIME PREDICTION PIPELINE")
    
    
    # STEP 1: LOAD DATA
    
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    print(f"   Features:  {X_train.shape[1]}")
    
    # STEP 2: PREPROCESSING
    
    X_train_prep, X_test_prep, preprocessor = preprocess_data(
        X_train, 
        X_test,
        handle_outliers=config.HANDLE_OUTLIERS
    )
    
    print(f"  Preprocessing completed")
    print(f"   Train shape: {X_train_prep.shape}")
    print(f"   Test shape:  {X_test_prep.shape}")
    
    # Save preprocessor
    preprocessor_path = config.MODEL_DIR / config.PREPROCESSOR_NAME
    preprocessor.save(preprocessor_path)
    
    # STEP 3: FEATURE ENGINEERING
    
    
    X_train_final, X_test_final, engineer = engineer_features(
        X_train_prep, 
        X_test_prep
    )
    
    print(f"   Train shape: {X_train_final.shape}")
    print(f"   Test shape:  {X_test_final.shape}")
    print(f"   Features created: {len(engineer.get_created_features())}")
    
    # Save feature engineer
    engineer_path = config.MODEL_DIR / config.FEATURE_ENGINEER_NAME
    engineer.save(engineer_path)
    print(f"    Feature engineer saved to: {engineer_path}")
    
    # STEP 4: TRAIN MODELS

    models, comparison = train_multiple_models(
        X_train_final, y_train,
        X_test_final, y_test,
        feature_names=engineer.get_feature_names(),
        save_models=True
    )
    
    
    # STEP 5: SELECT BEST MODEL
      
    best_name, best_model = get_best_model(comparison, models, metric='rmse')
    
    # Save best model with standard name
    best_model_path = config.MODEL_DIR / config.MODEL_NAME
    save_model(best_model, best_model_path)
    
    # FINAL SUMMARY
       
    print(f"\n BEST MODEL: {best_name.upper()}")
    print(f"\n PERFORMANCE METRICS:")
    best_metrics = comparison.loc[best_name]
    print(f"   RMSE:  {best_metrics['rmse']:.4f}")
    print(f"   MAE:   {best_metrics['mae']:.4f}")
    print(f"   R²:    {best_metrics['r2']:.4f}")
    print(f"   MAPE:  {best_metrics['mape']:.4f}%")
    
    print(f" SAVED FILES:")
    print(f"   Model:            {best_model_path}")
    print(f"   Preprocessor:     {preprocessor_path}")
    print(f"   Feature Engineer: {engineer_path}")
    
    print(f" MODEL COMPARISON:")
    print(comparison.to_string())
    
    print("\n NEXT STEPS:")
    print("   1. Check reports/ for detailed analysis")
    print("   2. Use predictor.py to make predictions on new orders")
    print("   3. Review feature importance in explainability.md")
    
    print("\n" + "="*80 + "\n")
    
    return best_model, preprocessor, engineer, comparison


if __name__ == "__main__":
    try:
        best_model, preprocessor, engineer, comparison = main()
        print(" Pipeline executed successfully!")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()