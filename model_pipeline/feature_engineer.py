"""
Feature Engineering Module
Creates new features from existing ones to improve model performance.
Based on domain knowledge and research insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

from . import config
from .utils import setup_logger, save_model, load_model, print_section_header


# Setup logger
logger = setup_logger(__name__, config.TRAINING_LOG_PATH)


class FeatureEngineer:
    """
    Feature Engineer for food delivery time prediction.
    Creates interaction features, polynomial features, and domain-specific features.
    
    Attributes:
        interaction_features: List of feature pairs to create interactions
        created_features: List of newly created feature names
        is_fitted: Whether the feature engineer has been fitted
        feature_names_in: Original feature names
        feature_names_out: Feature names after engineering
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> X_train_eng = engineer.fit_transform(X_train)
        >>> X_test_eng = engineer.transform(X_test)
    """
    
    def __init__(self, 
                 create_interactions: bool = None,
                 interaction_features: List[Tuple] = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            create_interactions: Whether to create interaction features (uses config if None)
            interaction_features: List of feature pairs for interactions (uses config if None)
        """
        self.create_interactions = create_interactions if create_interactions is not None else config.CREATE_INTERACTIONS
        self.interaction_features = interaction_features or config.INTERACTION_FEATURES
        
        self.created_features = []
        self.feature_names_in = []
        self.feature_names_out = []
        self.is_fitted = False
        
        # Store column mappings for categorical interactions
        self.categorical_columns_map = {}
        
        logger.info(f"FeatureEngineer initialized (interactions: {self.create_interactions})")
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        Args:
            X: Training features (after preprocessing/encoding)
            y: Training target (not used, for sklearn compatibility)
        
        Returns:
            Self
        """
        print_section_header("FITTING FEATURE ENGINEER")
        
        self.feature_names_in = X.columns.tolist()
        logger.info(f"Fitting feature engineer on {len(self.feature_names_in)} features")
        
        # Identify encoded categorical columns
        self._identify_categorical_columns(X)
        
        self.is_fitted = True
        logger.info("Feature engineer fitted successfully")
        print("Feature engineer fitted successfully\n")
        
        return self
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating new features.
        
        Args:
            X: Features to transform (preprocessed)
        
        Returns:
            Transformed features with new features added
        
        Raises:
            RuntimeError: If feature engineer hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform. Call fit() first.")
        
        logger.info(f"Transforming data with shape: {X.shape}")
        
        X_engineered = X.copy()
        self.created_features = []
        
        # 1. Create interaction features
        if self.create_interactions:
            X_engineered = self._create_interaction_features(X_engineered)
        
        # 2. Create domain-specific features
        X_engineered = self._create_domain_features(X_engineered)
        
        # 3. Create statistical features
        X_engineered = self._create_statistical_features(X_engineered)
        
        # Update output feature names
        self.feature_names_out = X_engineered.columns.tolist()
        
        logger.info(f"Transformed data shape: {X_engineered.shape}")
        logger.info(f"Created {len(self.created_features)} new features")
        
        return X_engineered
    
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit feature engineer and transform data in one step.
        
        Args:
            X: Training features
            y: Training target (not used)
        
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    
    def _identify_categorical_columns(self, X: pd.DataFrame) -> None:
        """
        Identify which columns are encoded categoricals (from OneHot encoding).
        Stores mapping of original categorical name to encoded column names.
        """
        # If using OneHot encoding, columns will have pattern: "OriginalName_Category"
        if config.ENCODING_STRATEGY == "onehot":
            for cat_feature in config.CATEGORICAL_FEATURES:
                # Find all columns that start with this feature name
                encoded_cols = [col for col in X.columns if col.startswith(f"{cat_feature}_")]
                if encoded_cols:
                    self.categorical_columns_map[cat_feature] = encoded_cols
                    logger.info(f"Identified {len(encoded_cols)} columns for {cat_feature}")
    
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features based on config.
        Handles both numerical and categorical (encoded) features.
        """
        print("🔧 Creating interaction features...")
        X_new = X.copy()
        n_created = 0
        
        for feature1, feature2 in self.interaction_features:
            # Check if features exist in the data
            feature1_exists = feature1 in X.columns or feature1 in self.categorical_columns_map
            feature2_exists = feature2 in X.columns or feature2 in self.categorical_columns_map
            
            if not feature1_exists or not feature2_exists:
                logger.warning(f"Cannot create interaction: {feature1} x {feature2} (features not found)")
                continue
            
            # Case 1: Both are numerical
            if feature1 in X.columns and feature2 in X.columns:
                new_feature_name = f"{feature1}_x_{feature2}"
                X_new[new_feature_name] = X[feature1] * X[feature2]
                self.created_features.append(new_feature_name)
                n_created += 1
            
            # Case 2: feature1 is numerical, feature2 is categorical (encoded)
            elif feature1 in X.columns and feature2 in self.categorical_columns_map:
                for cat_col in self.categorical_columns_map[feature2]:
                    new_feature_name = f"{feature1}_x_{cat_col}"
                    X_new[new_feature_name] = X[feature1] * X[cat_col]
                    self.created_features.append(new_feature_name)
                    n_created += 1
            
            # Case 3: feature1 is categorical, feature2 is numerical
            elif feature1 in self.categorical_columns_map and feature2 in X.columns:
                for cat_col in self.categorical_columns_map[feature1]:
                    new_feature_name = f"{cat_col}_x_{feature2}"
                    X_new[new_feature_name] = X[cat_col] * X[feature2]
                    self.created_features.append(new_feature_name)
                    n_created += 1
            
            # Case 4: Both are categorical (encoded) - create all pairwise interactions
            elif feature1 in self.categorical_columns_map and feature2 in self.categorical_columns_map:
                for cat_col1 in self.categorical_columns_map[feature1]:
                    for cat_col2 in self.categorical_columns_map[feature2]:
                        new_feature_name = f"{cat_col1}_x_{cat_col2}"
                        X_new[new_feature_name] = X[cat_col1] * X[cat_col2]
                        self.created_features.append(new_feature_name)
                        n_created += 1
        
        print(f"   ✓ Created {n_created} interaction features")
        logger.info(f"Created {n_created} interaction features")
        
        return X_new
    
    
    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features based on delivery logistics knowledge.
        """
        print("🔧 Creating domain-specific features...")
        X_new = X.copy()
        n_created = 0
        
        # Feature 1: Distance per preparation time (efficiency metric)
        if 'Distance_km' in X.columns and 'Preparation_Time_min' in X.columns:
            # Avoid division by zero
            X_new['Distance_per_PrepTime'] = X['Distance_km'] / (X['Preparation_Time_min'] + 1)
            self.created_features.append('Distance_per_PrepTime')
            n_created += 1
        
        # Feature 2: Experience level categories (binned)
        if 'Courier_Experience_yrs' in X.columns:
            # Create binary features for experience levels
            X_new['Is_Junior_Courier'] = (X['Courier_Experience_yrs'] <= 2).astype(int)
            X_new['Is_Senior_Courier'] = (X['Courier_Experience_yrs'] >= 5).astype(int)
            self.created_features.extend(['Is_Junior_Courier', 'Is_Senior_Courier'])
            n_created += 2
        
        # Feature 3: Distance categories (short/medium/long)
        if 'Distance_km' in X.columns:
            X_new['Is_Short_Distance'] = (X['Distance_km'] <= 5).astype(int)
            X_new['Is_Long_Distance'] = (X['Distance_km'] >= 15).astype(int)
            self.created_features.extend(['Is_Short_Distance', 'Is_Long_Distance'])
            n_created += 2
        
        # Feature 4: High preparation time flag
        if 'Preparation_Time_min' in X.columns:
            X_new['Is_High_PrepTime'] = (X['Preparation_Time_min'] >= 20).astype(int)
            self.created_features.append('Is_High_PrepTime')
            n_created += 1
        
        # Feature 5: Total time proxy (distance + preparation)
        if 'Distance_km' in X.columns and 'Preparation_Time_min' in X.columns:
            # Rough estimate: 2 min per km + prep time
            X_new['Estimated_Base_Time'] = (X['Distance_km'] * 2) + X['Preparation_Time_min']
            self.created_features.append('Estimated_Base_Time')
            n_created += 1
        
        # Feature 6: Adverse conditions (combining weather and traffic)
        # This requires checking encoded columns
        if config.ENCODING_STRATEGY == "onehot":
            # Check for rainy or snowy weather
            rainy_cols = [col for col in X.columns if 'Weather_Rainy' in col or 'Weather_Snowy' in col]
            high_traffic_cols = [col for col in X.columns if 'Traffic_Level_High' in col]
            
            if rainy_cols and high_traffic_cols:
                # Create adverse condition flag
                is_bad_weather = X[rainy_cols].sum(axis=1) > 0
                is_high_traffic = X[high_traffic_cols].sum(axis=1) > 0
                X_new['Adverse_Conditions'] = (is_bad_weather | is_high_traffic).astype(int)
                self.created_features.append('Adverse_Conditions')
                n_created += 1
        
        print(f"   ✓ Created {n_created} domain-specific features")
        logger.info(f"Created {n_created} domain-specific features")
        
        return X_new
    
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from numerical columns.
        """
        print("🔧 Creating statistical features...")
        X_new = X.copy()
        n_created = 0
        
        # Get numerical columns (original ones, not encoded)
        numerical_cols = [col for col in config.NUMERICAL_FEATURES if col in X.columns]
        
        if len(numerical_cols) >= 2:
            # Feature 1: Ratio of Distance to Courier Experience
            if 'Distance_km' in X.columns and 'Courier_Experience_yrs' in X.columns:
                X_new['Distance_per_Experience'] = X['Distance_km'] / (X['Courier_Experience_yrs'] + 0.1)
                self.created_features.append('Distance_per_Experience')
                n_created += 1
            
            # Feature 2: Squared distance (non-linear effect)
            if 'Distance_km' in X.columns:
                X_new['Distance_km_squared'] = X['Distance_km'] ** 2
                self.created_features.append('Distance_km_squared')
                n_created += 1
            
            # Feature 3: Log of distance (for skewed distributions)
            if 'Distance_km' in X.columns:
                X_new['Distance_km_log'] = np.log1p(X['Distance_km'])
                self.created_features.append('Distance_km_log')
                n_created += 1
        
        print(f"   ✓ Created {n_created} statistical features")
        logger.info(f"Created {n_created} statistical features")
        
        return X_new
    
    
    def get_feature_names(self) -> List[str]:
        """
        Get all feature names after engineering.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted first")
        return self.feature_names_out.copy()
    
    
    def get_created_features(self) -> List[str]:
        """
        Get list of newly created feature names.
        
        Returns:
            List of created feature names
        """
        return self.created_features.copy()
    
    
    def get_feature_info(self) -> Dict:
        """
        Get information about feature engineering.
        
        Returns:
            Dictionary with feature engineering information
        """
        return {
            'n_features_in': len(self.feature_names_in),
            'n_features_out': len(self.feature_names_out),
            'n_created': len(self.created_features),
            'created_features': self.created_features.copy(),
            'interaction_features_enabled': self.create_interactions,
            'interaction_pairs': self.interaction_features
        }
    
    
    def save(self, filepath: Path) -> None:
        """
        Save feature engineer to disk.
        
        Args:
            filepath: Path where to save
        """
        save_model(self, filepath)
        logger.info(f"FeatureEngineer saved to {filepath}")
    
    
    @staticmethod
    def load(filepath: Path) -> 'FeatureEngineer':
        """
        Load feature engineer from disk.
        
        Args:
            filepath: Path to saved feature engineer
        
        Returns:
            Loaded feature engineer
        """
        engineer = load_model(filepath)
        logger.info(f"FeatureEngineer loaded from {filepath}")
        return engineer
    
    
    def summary(self) -> None:
        """Print summary of feature engineering."""
        print_section_header("FEATURE ENGINEER SUMMARY")
        
        print(f"Status: {'✅ Fitted' if self.is_fitted else '❌ Not fitted'}")
        
        if self.is_fitted:
            print(f"\n📊 Features:")
            print(f"   Input features:    {len(self.feature_names_in)}")
            print(f"   Output features:   {len(self.feature_names_out)}")
            print(f"   Created features:  {len(self.created_features)}")
            
            if self.create_interactions:
                print(f" Interaction Features:")
                print(f"   Enabled: Yes")
                print(f"   Pairs defined: {len(self.interaction_features)}")
                for f1, f2 in self.interaction_features:
                    print(f"      - {f1} × {f2}")
            else:
                print(f" Interaction Features: Disabled")
            
            if self.created_features:
                print(f"Created Features ({len(self.created_features)}):")
                # Group by type
                interaction_features = [f for f in self.created_features if '_x_' in f]
                domain_features = [f for f in self.created_features if f not in interaction_features]
                
                if interaction_features:
                    print(f"   Interactions: {len(interaction_features)}")
                if domain_features:
                    print(f"   Domain-specific: {len(domain_features)}")
                    for feat in domain_features[:10]:  # Show first 10
                        print(f"      - {feat}")
                    if len(domain_features) > 10:
                        print(f"      ... and {len(domain_features) - 10} more")
        
        print()



# CONVENIENCE FUNCTIONS


def engineer_features(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     create_interactions: bool = None) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureEngineer]:
    """
    Convenience function to engineer features for train and test data.
    
    Args:
        X_train: Training features (preprocessed)
        X_test: Test features (preprocessed)
        create_interactions: Whether to create interaction features (uses config if None)
    
    Returns:
        Tuple of (X_train_engineered, X_test_engineered, feature_engineer)
    
    Example:
        >>> X_train_eng, X_test_eng, engineer = engineer_features(X_train, X_test)
    """
    # Initialize feature engineer
    engineer = FeatureEngineer(create_interactions=create_interactions)
    
    # Fit and transform
    X_train_engineered = engineer.fit_transform(X_train)
    X_test_engineered = engineer.transform(X_test)
    
    # Print summary
    engineer.summary()
    
    logger.info(f"Features engineered: Train {X_train_engineered.shape}, Test {X_test_engineered.shape}")
    
    return X_train_engineered, X_test_engineered, engineer


# MAIN EXECUTION (for testing)

if __name__ == "__main__":
    """Test the FeatureEngineer class."""
    
    print("TESTING FEATURE ENGINEER MODULE")
    
    # Load and preprocess data
    from data_loader import get_train_test_data
    from preprocessor import preprocess_data
    
    X_train, X_test, y_train, y_test = get_train_test_data()
    X_train_prep, X_test_prep, preprocessor = preprocess_data(X_train, X_test)
    
    print(f"After preprocessing:")
    print(f"   X_train: {X_train_prep.shape}")
    print(f"   X_test:  {X_test_prep.shape}")
    
    # Engineer features
    X_train_eng, X_test_eng, engineer = engineer_features(
        X_train_prep, X_test_prep,
        create_interactions=True
    )
    
    print(f" After feature engineering:")
    print(f"   X_train: {X_train_eng.shape}")
    print(f"   X_test:  {X_test_eng.shape}")
    
    # Show feature info
    info = engineer.get_feature_info()
    print(f" Feature Engineering Results:")
    print(f"   Features added: {info['n_created']}")
    print(f"   Total features: {info['n_features_out']}")
    
    print(f" Sample of created features:")
    created = engineer.get_created_features()
    for feat in created[:15]:
        print(f"      - {feat}")
    if len(created) > 15:
        print(f"      ... and {len(created) - 15} more")
    
    print("FEATURE ENGINEER TEST COMPLETED")
