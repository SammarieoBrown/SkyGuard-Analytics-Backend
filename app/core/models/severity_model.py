import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import random

from app.config import SEVERITY_MODEL_PATH

logger = logging.getLogger(__name__)


class MockSeverityModel:
    """Mock model for development when real model can't be loaded"""
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return realistic mock severity predictions"""
        n_samples = len(features) if hasattr(features, '__len__') else 1
        severity_levels = ['Minor', 'Moderate', 'Significant', 'Severe', 'Catastrophic']
        predictions = []
        for _ in range(n_samples):
            # Weighted random selection favoring lower severities
            weights = [0.4, 0.3, 0.2, 0.08, 0.02]
            prediction = random.choices(severity_levels, weights=weights)[0]
            predictions.append(prediction)
        return np.array(predictions)


class SeverityModel:
    """
    Wrapper for the severity prediction model.
    Loads and uses the model to predict severity levels of weather events.
    """
    
    # Severity descriptions from the original model
    SEVERITY_DESCRIPTIONS = {
        'Minor': 'Low impact, localized effects. Minimal property damage and no casualties expected.',
        'Moderate': 'Notable damage in affected areas. Some injuries possible, limited property damage.',
        'Significant': 'Major damage expected. Multiple casualties likely, substantial property losses.',
        'Severe': 'Widespread damage across region. Many casualties expected, extensive property damage.',
        'Catastrophic': 'Extreme impact with disaster declaration likely. Mass casualties and devastating damage.'
    }
    
    # Severity color mappings for visualization
    SEVERITY_COLORS = {
        'Minor': '#92c5de',        # Light blue
        'Moderate': '#4393c3',     # Medium blue
        'Significant': '#f4a582',  # Light orange
        'Severe': '#d6604d',       # Orange-red
        'Catastrophic': '#b2182b'  # Dark red
    }
    
    def __init__(self):
        """Initialize the severity model."""
        self.model = None
        self.is_mock = False
        self.severity_levels = ['Minor', 'Moderate', 'Significant', 'Severe', 'Catastrophic']
        model_path = Path(SEVERITY_MODEL_PATH)
        
        try:
            self.model = self._load_model(model_path)
            logger.info("Severity model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load real model, using mock: {str(e)}")
            self.model = MockSeverityModel()
            self.is_mock = True
        
    def _load_model(self, model_path: Path):
        """
        Load the model from the pickle file.
        
        Args:
            model_path (Path): Path to the model file
            
        Returns:
            The loaded model object
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Try loading with joblib first
            model = joblib.load(model_path)
            logger.info("Severity model loaded successfully with joblib")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model with joblib: {str(e)}. Trying pickle...")
            
            try:
                # Fallback to pickle
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                logger.info("Severity model loaded successfully with pickle")
                return model
            except Exception as e:
                logger.error(f"Failed to load severity model: {str(e)}")
                raise RuntimeError(f"Failed to load severity model: {str(e)}")
    
    def create_dataframe(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a dataframe from the input parameters that the model can use.
        
        Args:
            params (Dict[str, Any]): Input parameters for prediction
            
        Returns:
            pd.DataFrame: DataFrame formatted for model input
        """
        import datetime
        
        # Extract base parameters
        event_type = params.get('event_type', 'Unknown')
        state = params.get('state', 'TX')
        magnitude = params.get('magnitude', 0.0)
        duration_hours = params.get('duration_hours', 1.0)
        property_damage = params.get('property_damage', 0)
        crop_damage = params.get('crop_damage', 0)
        injuries = params.get('injuries', 0)
        deaths = params.get('deaths', 0)
        month = params.get('month', 6)
        hour = params.get('hour', 12)
        tor_f_scale = params.get('tor_f_scale', '')
        cz_type = params.get('cz_type', 'Z')
        
        # Create engineered features
        casualties_total = injuries + deaths
        has_magnitude = 1 if magnitude > 0 else 0
        
        # Tornado features
        is_tornado = 1 if 'tornado' in event_type.lower() else 0
        tornado_scale = 0
        if tor_f_scale:
            scale_map = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4, 'F5': 5,
                         'EF0': 0, 'EF1': 1, 'EF2': 2, 'EF3': 3, 'EF4': 4, 'EF5': 5}
            tornado_scale = scale_map.get(tor_f_scale, 0)
        
        # High severity event types
        high_severity_types = ['tornado', 'hurricane', 'flood', 'wildfire', 'storm surge']
        is_high_severity_type = 1 if any(t in event_type.lower() for t in high_severity_types) else 0
        
        # Time features
        current_date = datetime.datetime.now()
        day_of_week = current_date.weekday()
        year = current_date.year
        is_night = 1 if hour < 6 or hour >= 18 else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        is_summer = 1 if month in [6, 7, 8] else 0
        is_winter = 1 if month in [12, 1, 2] else 0
        
        # Cyclical time features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # High risk states (tornado alley, hurricane zones)
        high_risk_states = ['OK', 'TX', 'KS', 'FL', 'LA', 'AL', 'MS', 'GA', 'SC', 'NC']
        is_high_risk_state = 1 if state in high_risk_states else 0
        
        # Composite features
        impact_score = np.log1p(property_damage) + np.log1p(crop_damage) + casualties_total * 10
        high_magnitude_long_duration = 1 if magnitude > 50 and duration_hours > 6 else 0
        casualties_property_damage = casualties_total * np.log1p(property_damage)
        night_high_severity = is_night * is_high_severity_type
        
        # Create feature dictionary in the expected order
        features = {
            'damage_property_num': property_damage,
            'damage_crops_num': crop_damage,
            'injuries_total': injuries,
            'deaths_total': deaths,
            'casualties_total': casualties_total,
            'magnitude': magnitude,
            'has_magnitude': has_magnitude,
            'event_duration_hours': duration_hours,
            'tornado_scale': tornado_scale,
            'is_tornado': is_tornado,
            'is_high_severity_type': is_high_severity_type,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'is_night': is_night,
            'is_weekend': is_weekend,
            'is_summer': is_summer,
            'is_winter': is_winter,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'is_high_risk_state': is_high_risk_state,
            'event_type': event_type,
            'state': state,
            'cz_type': cz_type,
            'impact_score': impact_score,
            'high_magnitude_long_duration': high_magnitude_long_duration,
            'casualties_property_damage': casualties_property_damage,
            'night_high_severity': night_high_severity
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # If we have a preprocessor, use it to encode categorical features
        if hasattr(self.model, 'preprocessor') and hasattr(self.model.preprocessor, 'label_encoders'):
            # Encode categorical features
            if 'event_type' in self.model.preprocessor.label_encoders:
                le = self.model.preprocessor.label_encoders['event_type']
                df['event_type_encoded'] = df['event_type'].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                df['event_type_encoded'] = 0
                
            if 'state' in self.model.preprocessor.label_encoders:
                le = self.model.preprocessor.label_encoders['state']
                df['state_encoded'] = df['state'].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                df['state_encoded'] = 0
                
            if 'cz_type' in self.model.preprocessor.label_encoders:
                le = self.model.preprocessor.label_encoders['cz_type']
                df['cz_type_encoded'] = df['cz_type'].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                df['cz_type_encoded'] = 0
        else:
            # Simple encoding if no preprocessor
            df['event_type_encoded'] = 0
            df['state_encoded'] = 0  
            df['cz_type_encoded'] = 0
            
        # Drop the original categorical columns
        df = df.drop(['event_type', 'state', 'cz_type'], axis=1)
        
        # Ensure columns are in the right order if we have feature_columns
        if hasattr(self.model, 'feature_columns'):
            # Reorder columns to match expected order
            ordered_df = pd.DataFrame()
            for col in self.model.feature_columns:
                if col in df.columns:
                    ordered_df[col] = df[col]
                else:
                    # Add missing columns with default values
                    ordered_df[col] = 0
            df = ordered_df
            
        return df
    
    def predict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a severity prediction based on input parameters.
        
        Args:
            params (Dict[str, Any]): Input parameters for prediction
            
        Returns:
            Dict[str, Any]: Prediction results including severity level and description
        """
        try:
            # Create dataframe from parameters
            df = self.create_dataframe(params)
            
            # Make prediction
            if hasattr(self.model, "predict_proba"):
                # For models with probability estimates, get the class probabilities
                probabilities = self.model.predict_proba(df)[0]
                severity_class = self.model.predict(df)[0]
                
                # Map numeric classes to severity names
                severity_names = ['Minor', 'Moderate', 'Significant', 'Severe', 'Catastrophic']
                if hasattr(self.model, "classes_"):
                    class_names = self.model.classes_
                    # Create probability distribution by severity name
                    class_probs = {}
                    for idx, prob in zip(class_names, probabilities):
                        if 0 <= idx < len(severity_names):
                            class_probs[severity_names[idx]] = float(prob)
                        else:
                            class_probs[f"class_{idx}"] = float(prob)
                else:
                    class_probs = {severity_names[i]: float(prob) for i, prob in enumerate(probabilities) if i < len(severity_names)}
            else:
                # For models without probability estimates
                severity_class = self.model.predict(df)[0]
                # Create a simulated high confidence
                class_probs = {str(severity_class): 0.95}
            
            # Map numeric class to severity name
            severity_names = ['Minor', 'Moderate', 'Significant', 'Severe', 'Catastrophic']
            if isinstance(severity_class, (int, np.integer)):
                if 0 <= severity_class < len(severity_names):
                    severity_class = severity_names[severity_class]
                else:
                    severity_class = 'Unknown'
            else:
                # Try to extract numeric value from string like "[4]"
                severity_str = str(severity_class)
                if severity_str.startswith('[') and severity_str.endswith(']'):
                    try:
                        numeric_class = int(severity_str[1:-1])
                        if 0 <= numeric_class < len(severity_names):
                            severity_class = severity_names[numeric_class]
                        else:
                            severity_class = 'Unknown'
                    except:
                        severity_class = 'Unknown'
            
            # Get description and color for the predicted severity
            description = self.SEVERITY_DESCRIPTIONS.get(
                severity_class, 
                "Severity level description not available"
            )
            color = self.SEVERITY_COLORS.get(
                severity_class, 
                "#808080"  # Default to gray if no matching color
            )
            
            # Calculate confidence score (simplified)
            confidence_score = class_probs.get(severity_class, 0.8)
            
            # Get potential impact factors
            impact_factors = self._get_impact_factors(params)
            
            return {
                "severity_class": severity_class,
                "description": description,
                "color_code": color,
                "confidence_score": float(confidence_score),
                "class_probabilities": class_probs,
                "impact_factors": impact_factors
            }
        except Exception as e:
            logger.error(f"Error during severity prediction: {str(e)}")
            raise
    
    def _get_impact_factors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get impact factors based on the event parameters.
        
        Args:
            params (Dict[str, Any]): Input parameters for prediction
            
        Returns:
            Dict[str, Any]: Impact factors for the event
        """
        impact = {}
        
        # Add property damage impact
        if params.get('property_damage', 0) > 0:
            impact['property_damage'] = {
                'value': params['property_damage'],
                'impact_level': self._assess_damage_level(params['property_damage'])
            }
            
        # Add crop damage impact
        if params.get('crop_damage', 0) > 0:
            impact['crop_damage'] = {
                'value': params['crop_damage'],
                'impact_level': self._assess_damage_level(params['crop_damage'])
            }
            
        # Add casualty impact
        casualties = params.get('injuries', 0) + params.get('deaths', 0)
        if casualties > 0:
            impact['casualties'] = {
                'injuries': params.get('injuries', 0),
                'deaths': params.get('deaths', 0),
                'total': casualties,
                'impact_level': self._assess_casualty_level(casualties)
            }
            
        # Add event-specific impact factors
        if params.get('event_type') == 'Tornado' and params.get('tor_f_scale'):
            impact['tornado_scale'] = {
                'value': params['tor_f_scale'],
                'impact_level': self._assess_tornado_level(params['tor_f_scale'])
            }
            
        return impact
    
    def _assess_damage_level(self, damage: float) -> str:
        """
        Assess damage level category based on dollar amount.
        
        Args:
            damage (float): Dollar amount of damage
            
        Returns:
            str: Damage level category
        """
        if damage < 10000:
            return "Low"
        elif damage < 100000:
            return "Moderate"
        elif damage < 1000000:
            return "High"
        else:
            return "Extreme"
    
    def _assess_casualty_level(self, casualties: int) -> str:
        """
        Assess casualty level based on count.
        
        Args:
            casualties (int): Number of casualties (injuries + deaths)
            
        Returns:
            str: Casualty level category
        """
        if casualties < 5:
            return "Low"
        elif casualties < 20:
            return "Moderate"
        elif casualties < 50:
            return "High"
        else:
            return "Extreme"
    
    def _assess_tornado_level(self, tor_f_scale: str) -> str:
        """
        Assess tornado impact level based on F/EF scale.
        
        Args:
            tor_f_scale (str): Tornado F/EF scale
            
        Returns:
            str: Tornado impact level category
        """
        scale_map = {
            'F0': 'Low',
            'F1': 'Low',
            'EF0': 'Low',
            'EF1': 'Low',
            'F2': 'Moderate',
            'EF2': 'Moderate', 
            'F3': 'High',
            'EF3': 'High',
            'F4': 'Extreme',
            'EF4': 'Extreme',
            'F5': 'Extreme',
            'EF5': 'Extreme'
        }
        return scale_map.get(tor_f_scale, 'Unknown') 