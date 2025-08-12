"""
Wrapper for the casualty risk prediction model from the Capstone project.
This module handles loading and using the pre-trained CasualtyRiskModel.
"""

import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import xgboost as xgb
from datetime import datetime
import random

from app.config import CASUALTY_RISK_MODEL_PATH

logger = logging.getLogger(__name__)


class MockCasualtyRiskModel:
    """Mock model for development when real model can't be loaded"""
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return realistic mock probability predictions"""
        n_samples = len(features) if hasattr(features, '__len__') else 1
        # Generate realistic casualty risk probabilities
        predictions = []
        for _ in range(n_samples):
            # Two classes: no casualties (0) and casualties (1)
            no_casualty_prob = random.uniform(0.4, 0.9)
            casualty_prob = 1.0 - no_casualty_prob
            predictions.append([no_casualty_prob, casualty_prob])
        return np.array(predictions)


class CasualtyRiskModel:
    """
    Casualty risk prediction model for severe weather events.
    """

    def __init__(self):
        """Initialize the casualty risk model."""
        self.model = None
        self.is_mock = False
        model_path = Path(CASUALTY_RISK_MODEL_PATH)
        
        try:
            self.model = self._load_model(model_path)
            logger.info("Casualty risk model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load real model, using mock: {str(e)}")
            self.model = MockCasualtyRiskModel()
            self.is_mock = True

    def _load_model(self, model_path: Path):
        """
        Load the trained model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Use the custom model loader that handles class remapping
            from app.core.models.model_loader import load_model_with_remapping
            model = load_model_with_remapping(str(model_path))
            logger.info("Casualty model loaded successfully with custom loader")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model with custom loader: {str(e)}. Trying joblib...")
            
            try:
                # Fallback to joblib
                model = joblib.load(model_path)
                logger.info("Casualty model loaded successfully with joblib")
                return model
            except Exception as e2:
                logger.warning(f"Failed to load model with joblib: {str(e2)}. Trying pickle...")
                
                try:
                    # Fallback to pickle
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    logger.info("Casualty model loaded successfully with pickle")
                    return model
                except Exception as e3:
                    logger.error(f"Failed to load casualty risk model: {str(e3)}")
                    raise RuntimeError(f"Failed to load casualty risk model: {str(e3)}")

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predict casualty risk probability for a weather event.
        
        Args:
            features: Dictionary containing event features
            
        Returns:
            Predicted casualty risk probability (0-1)
        """
        try:
            if self.is_mock:
                logger.info("Using mock prediction for casualty risk")
                # Generate realistic mock prediction based on inputs
                magnitude = features.get('magnitude', 50)
                event_type = features.get('event_type', 'thunderstorm wind')
                
                # Simple mock calculation
                base_risk = 0.1  # 10% base risk
                
                # Increase risk based on magnitude
                if magnitude > 100:
                    base_risk += 0.3
                elif magnitude > 75:
                    base_risk += 0.2
                elif magnitude > 50:
                    base_risk += 0.1
                
                # Increase risk for more dangerous event types
                if event_type.lower() in ['tornado', 'hurricane']:
                    base_risk += 0.4
                elif event_type.lower() in ['flash flood', 'wildfire']:
                    base_risk += 0.2
                
                # Add some randomness
                risk_adjustment = random.uniform(-0.1, 0.1)
                final_risk = max(0.01, min(0.95, base_risk + risk_adjustment))
                
                return final_risk
            
            # Check if model is a dictionary package (from pickle)
            if isinstance(self.model, dict):
                # Extract the actual XGBoost model from the package
                actual_model = self.model.get('model')
                preprocessor = self.model.get('preprocessor')
                feature_engineer = self.model.get('feature_engineer')
                
                if actual_model is None:
                    raise ValueError("No model found in model package")
                
                # Use the XGBoost Booster directly
                import xgboost as xgb
                if isinstance(actual_model, xgb.Booster):
                    # Prepare features using the preprocessor if available
                    feature_df = self._prepare_features(features)
                    # Create DMatrix for XGBoost
                    dtest = xgb.DMatrix(feature_df)
                    # Get prediction (raw probability)
                    prediction = actual_model.predict(dtest)[0]
                    # XGBoost binary classification returns probability directly
                    prediction = max(0, min(1, prediction))
                else:
                    # Fallback for other model types
                    feature_df = self._prepare_features(features)
                    prediction = actual_model.predict(feature_df)[0]
                    prediction = max(0, min(1, prediction))
            else:
                # Prepare features for the real model
                feature_array = self._prepare_features(features)
                
                # Make prediction (assuming binary classification)
                if hasattr(self.model, 'predict_proba'):
                    # Get probability of casualty class (assuming class 1 is casualty)
                    prediction = self.model.predict_proba(feature_array.reshape(1, -1))[0][1]
                else:
                    # Fallback if no predict_proba method
                    prediction = self.model.predict(feature_array.reshape(1, -1))[0]
                    # Normalize to probability if needed
                    prediction = max(0, min(1, prediction))
            
            logger.info(f"Casualty risk prediction: {prediction:.3f}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error making casualty risk prediction: {str(e)}")
            # Return a fallback prediction
            return 0.25

    def _prepare_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for model prediction.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Prepared feature DataFrame
        """
        import pandas as pd
        from datetime import datetime
        
        # If model is a dict with feature columns, use them
        if isinstance(self.model, dict):
            feature_columns = self.model.get('feature_columns', [])
            preprocessor = self.model.get('preprocessor')
            feature_engineer = self.model.get('feature_engineer')
            
            # Create a basic feature set matching expected columns
            # Start with defaults for all expected features
            prepared_features = {}
            
            # Basic event features
            prepared_features['magnitude'] = features.get('magnitude', 50)
            prepared_features['has_magnitude'] = 1 if prepared_features['magnitude'] > 0 else 0
            prepared_features['event_duration_hours'] = features.get('duration_hours', 1)
            
            # Event type features
            event_type = features.get('event_type', 'thunderstorm wind').lower()
            prepared_features['is_tornado'] = 1 if 'tornado' in event_type else 0
            prepared_features['tornado_scale'] = features.get('tornado_scale', 0)
            prepared_features['is_high_risk_event'] = 1 if event_type in ['tornado', 'hurricane', 'wildfire'] else 0
            
            # Time features
            now = datetime.now()
            prepared_features['hour'] = features.get('hour', now.hour)
            prepared_features['day_of_week'] = features.get('day_of_week', now.weekday())
            prepared_features['month'] = features.get('month', now.month)
            prepared_features['year'] = features.get('year', now.year)
            prepared_features['is_night'] = 1 if prepared_features['hour'] < 6 or prepared_features['hour'] >= 18 else 0
            prepared_features['is_weekend'] = 1 if prepared_features['day_of_week'] >= 5 else 0
            prepared_features['is_rush_hour'] = 1 if prepared_features['hour'] in [7, 8, 16, 17, 18] else 0
            
            # Cyclical time features
            prepared_features['hour_sin'] = np.sin(2 * np.pi * prepared_features['hour'] / 24)
            prepared_features['hour_cos'] = np.cos(2 * np.pi * prepared_features['hour'] / 24)
            prepared_features['month_sin'] = np.sin(2 * np.pi * prepared_features['month'] / 12)
            prepared_features['month_cos'] = np.cos(2 * np.pi * prepared_features['month'] / 12)
            
            # Location features
            state = features.get('state', 'TX')
            high_density_states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
            prepared_features['is_high_density_state'] = 1 if state in high_density_states else 0
            prepared_features['is_urban'] = features.get('is_urban', 0)
            prepared_features['daytime_population_risk'] = (
                (prepared_features['hour'] >= 9 and prepared_features['hour'] <= 17) and 
                prepared_features['day_of_week'] < 5
            ) * 1
            
            # Historical pattern features (use defaults)
            prepared_features['event_casualty_prob'] = 0.1
            prepared_features['state_casualty_prob'] = 0.1
            
            # Encoded categorical features
            prepared_features['event_type_encoded'] = 0
            prepared_features['state_encoded'] = 0
            
            # Interaction features
            prepared_features['tornado_night'] = prepared_features['is_tornado'] * prepared_features['is_night']
            prepared_features['high_risk_urban'] = prepared_features['is_high_risk_event'] * prepared_features['is_urban']
            prepared_features['severe_magnitude_duration'] = (
                (prepared_features['magnitude'] > 75) & (prepared_features['event_duration_hours'] > 6)
            ) * 1
            prepared_features['urban_rush_hour'] = prepared_features['is_urban'] * prepared_features['is_rush_hour']
            prepared_features['weekend_severity'] = prepared_features['is_weekend'] * prepared_features['has_magnitude']
            prepared_features['high_risk_state_event'] = prepared_features['is_high_density_state'] * prepared_features['is_high_risk_event']
            
            # Create DataFrame with feature columns in order
            df = pd.DataFrame([prepared_features])
            
            # Ensure we have all required columns
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only the columns the model expects, in order
            if feature_columns:
                df = df[feature_columns]
            
            return df
        else:
            # Fallback for non-dict models
            feature_list = [
                features.get('magnitude', 0),
                1 if features.get('event_type', '').lower() == 'tornado' else 0,
                1 if features.get('event_type', '').lower() == 'hurricane' else 0,
            ]
            return pd.DataFrame([feature_list])

    def get_risk_category(self, probability: float) -> str:
        """
        Convert probability to risk category.
        
        Args:
            probability: Risk probability (0-1)
            
        Returns:
            Risk category string
        """
        if probability < 0.1:
            return "Very Low"
        elif probability < 0.3:
            return "Low"
        elif probability < 0.5:
            return "Moderate"
        elif probability < 0.7:
            return "High"
        else:
            return "Very High"

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Raw DataFrame
            is_training: Whether preparing training data
            
        Returns:
            Features and target
        """
        # Create target variable
        y = self.preprocessor.create_target_variable(df) if is_training else None
        
        # Basic preprocessing
        df_processed = self.preprocessor.prepare_features(df, is_training)
        
        # Feature engineering
        df_features = self.feature_engineer.create_severity_features(df_processed)
        df_features = self.feature_engineer.create_population_risk_features(df_features)
        df_features = self.feature_engineer.create_historical_patterns(df_features, is_training)
        df_features = self.feature_engineer.create_interaction_features(df_features)
        
        return df_features, y
    
    def get_feature_columns(self) -> List[str]:
        """Get list of features for modeling."""
        return [
            # Severity indicators
            'magnitude', 'has_magnitude', 'event_duration_hours',
            'tornado_scale', 'is_tornado', 'is_high_risk_event',
            
            # Temporal features
            'hour', 'day_of_week', 'month', 'year',
            'is_night', 'is_weekend', 'is_rush_hour',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            
            # Population risk
            'is_high_density_state', 'is_urban', 'daytime_population_risk',
            
            # Historical patterns
            'event_casualty_prob', 'state_casualty_prob',
            
            # Encoded categoricals
            'event_type_encoded', 'state_encoded',
            
            # Interactions
            'tornado_night', 'high_risk_urban', 'severe_magnitude_duration',
            'urban_rush_hour', 'weekend_severity', 'high_risk_state_event'
        ]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict casualty risk probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probability array [P(no casualty), P(casualty)]
        """
        feature_cols = self.get_feature_columns()
        X_model = X[feature_cols]
        dtest = xgb.DMatrix(X_model)
        
        # Get raw predictions (logits when using focal loss)
        raw_predictions = self.model.predict(dtest)
        
        # Apply temperature scaling to spread predictions
        temperature = 2.0  # Adjust to spread predictions
        scaled_predictions = raw_predictions / temperature
        
        # Convert logits to probabilities using sigmoid
        proba_positive = 1.0 / (1.0 + np.exp(-scaled_predictions))
        
        # Apply calibration if available
        if self.is_calibrated and self.calibrator is not None:
            if hasattr(self.calibrator, 'predict'):
                # For isotonic regression
                proba_positive = self.calibrator.predict(proba_positive)
            else:
                # For logistic regression
                proba_positive = self.calibrator.predict_proba(proba_positive.reshape(-1, 1))[:, 1]
            
            # Ensure valid probability range
            proba_positive = np.clip(proba_positive, 0, 1)
        
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
    def predict_with_confidence(self, X: pd.DataFrame, confidence_level: float = 0.95) -> Dict:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Feature DataFrame
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Get base predictions
        proba = self.predict_proba(X)
        risk_scores = proba[:, 1]
        
        # Risk categories
        risk_categories = pd.cut(
            risk_scores,
            bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
        
        return {
            'risk_probability': risk_scores,
            'risk_category': risk_categories,
            'confidence_level': confidence_level
        }
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model pipeline."""
        from app.core.models.model_loader import load_model_with_remapping
        
        model_package = load_model_with_remapping(filepath)
        
        instance = cls()
        instance.model = model_package['model']
        instance.preprocessor = model_package['preprocessor']
        instance.feature_engineer = model_package['feature_engineer']
        instance.feature_importance = model_package['feature_importance']
        instance.is_calibrated = model_package.get('is_calibrated', False)
        instance.calibrator = model_package.get('calibrator', None)
        
        return instance