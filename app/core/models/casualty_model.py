import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
from datetime import datetime

from app.config import CASUALTY_RISK_MODEL_PATH
from app.core.preprocessing import CasualtyRiskPreprocessor, CasualtyFeatureEngineer, FocalLoss
from app.core.models.casualty_model_wrapper import CasualtyRiskModel as CasualtyRiskModelWrapper

logger = logging.getLogger(__name__)


class CasualtyRiskModel:
    """
    Wrapper for the casualty risk prediction model.
    Loads and uses the model to predict casualty risk from weather events.
    """
    
    def __init__(self, model_path: Path = CASUALTY_RISK_MODEL_PATH):
        """
        Initialize the model by loading it from the specified path.
        
        Args:
            model_path (Path): Path to the pickle file containing the trained model
        """
        self.model_wrapper = self._load_model(model_path)
        self.risk_thresholds = {
            "low": 0.3,
            "moderate": 0.5,
            "high": 0.7,
            "very_high": 0.85
        }
        logger.info(f"Successfully loaded casualty risk model from {model_path}")
        
    def _load_model(self, model_path: Path):
        """
        Load the model from the pickle file.
        
        Args:
            model_path (Path): Path to the model file
            
        Returns:
            The loaded model object
        """
        try:
            # Load the full model wrapper
            model_wrapper = CasualtyRiskModelWrapper.load_model(str(model_path))
            return model_wrapper
        except Exception as e:
            logger.error(f"Failed to load casualty risk model: {str(e)}")
            raise RuntimeError(f"Failed to load casualty risk model: {str(e)}")
    
    def create_dataframe(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a dataframe from the input parameters that the model can use.
        
        Args:
            params (Dict[str, Any]): Input parameters for prediction
            
        Returns:
            pd.DataFrame: DataFrame formatted for model input
        """
        # Create a single row dataframe with all required features
        df = pd.DataFrame([{
            'event_type': params['event_type'],
            'state': params['state'],
            'magnitude': params.get('magnitude', 0.0),
            'event_begin_time': datetime.now(),  # Use current time as default
            'event_end_time': datetime.now() + pd.Timedelta(hours=params.get('duration_hours', 1.0)),
            'month': params.get('month', datetime.now().month),
            'hour': params.get('hour', datetime.now().hour),
            'injuries_direct': 0,  # Dummy values for preprocessing
            'injuries_indirect': 0,
            'deaths_direct': 0,
            'deaths_indirect': 0
        }])
        
        # Add tornado F-scale if available
        if params.get('tor_f_scale'):
            df['tor_f_scale'] = params['tor_f_scale']
        else:
            df['tor_f_scale'] = None
            
        # Add county/zone name if available
        if params.get('cz_name'):
            df['cz_name'] = params['cz_name']
        else:
            df['cz_name'] = ''
            
        return df
    
    def predict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a casualty risk prediction based on input parameters.
        
        Args:
            params (Dict[str, Any]): Input parameters for prediction
            
        Returns:
            Dict[str, Any]: Prediction results including risk probabilities
        """
        try:
            # Create dataframe from parameters
            df = self.create_dataframe(params)
            
            # Prepare features using the model's preprocessors
            df_features, _ = self.model_wrapper.prepare_data(df, is_training=False)
            
            # Get prediction probabilities
            probs = self.model_wrapper.predict_proba(df_features)
            risk_score = probs[0, 1]  # Probability of casualties
            
            # Get predictions with confidence
            confidence_results = self.model_wrapper.predict_with_confidence(df_features)
            
            # Determine risk level
            risk_level = self._get_risk_level(risk_score)
            
            # Get population factors (simplified example)
            population_risk_factors = self._get_population_risk_factors(params)
            
            return {
                "casualty_risk_score": float(risk_score),
                "risk_level": risk_level,
                "risk_category": str(confidence_results['risk_category'][0]),
                "probability": {
                    "no_casualties": float(probs[0, 0]),
                    "casualties": float(probs[0, 1])
                },
                "population_risk_factors": population_risk_factors
            }
        except Exception as e:
            logger.error(f"Error during casualty risk prediction: {str(e)}")
            raise
    
    def _get_risk_level(self, risk_score: float) -> float:
        """
        Convert raw risk score to a standardized risk level (0-10).
        
        Args:
            risk_score (float): Raw risk probability
            
        Returns:
            float: Standardized risk level from 0-10
        """
        # Convert probability to a 0-10 scale
        return min(10, round(risk_score * 10, 1))
    
    def _get_risk_category(self, risk_level: float) -> str:
        """
        Get a categorical risk assessment based on the risk level.
        
        Args:
            risk_level (float): Risk level from 0-10
            
        Returns:
            str: Categorical risk assessment
        """
        if risk_level < 3:
            return "Low"
        elif risk_level < 5:
            return "Moderate"
        elif risk_level < 7:
            return "High"
        else:
            return "Very High"
    
    def _get_population_risk_factors(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Get population-based risk factors for the event.
        This is a simplified example and would be replaced with actual data.
        
        Args:
            params (Dict[str, Any]): Input parameters for prediction
            
        Returns:
            Dict[str, float]: Population risk factors
        """
        # This would typically be based on actual demographic/population data
        # Here we just return a simplified example
        return {
            "population_density": 0.7,
            "vulnerable_population": 0.5,
            "infrastructure_resilience": 0.6
        }