from typing import Dict, Any, Optional
import logging
from app.core.models.model_manager import model_manager
from app.utils.json_encoder import numpy_to_python

logger = logging.getLogger(__name__)


class ImpactForecastingService:
    """
    Service for impact forecasting including property damage predictions.
    This service acts as an intermediary between the API and the underlying models.
    """
    
    def __init__(self):
        """Initialize service with required models from model manager."""
        # Use the singleton model_manager to get already-loaded models
        # This avoids reloading models on every request
        self.property_damage_model = model_manager.get_property_damage_model()
        self.casualty_risk_model = model_manager.get_casualty_risk_model()
        self.severity_model = model_manager.get_severity_model()
        logger.info("ImpactForecastingService initialized with preloaded models")
    
    def predict_property_damage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict property damage for a weather event.
        
        Args:
            params (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        logger.info(f"Predicting property damage for event: {params['event_type']} in {params['state']}")
        
        try:
            # Use the property damage model to make a prediction
            prediction = self.property_damage_model.predict(params)
            
            # Convert numpy types to Python native types
            
            return numpy_to_python(prediction)
        except Exception as e:
            logger.error(f"Error in property damage prediction service: {str(e)}")
            raise
            
    def predict_casualty_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict casualty risk for a weather event.
        
        Args:
            params (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        logger.info(f"Predicting casualty risk for event: {params['event_type']} in {params['state']}")
        
        try:
            # Use the casualty risk model to make a prediction
            prediction = self.casualty_risk_model.predict(params)
            
            # Convert numpy types to Python native types
            
            return numpy_to_python(prediction)
        except Exception as e:
            logger.error(f"Error in casualty risk prediction service: {str(e)}")
            raise
            
    def predict_severity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict severity level for a weather event.
        
        Args:
            params (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        logger.info(f"Predicting severity level for event: {params['event_type']} in {params['state']}")
        
        try:
            # Use the severity model to make a prediction
            prediction = self.severity_model.predict(params)
            
            # Convert numpy types to Python native types
            
            return numpy_to_python(prediction)
        except Exception as e:
            logger.error(f"Error in severity prediction service: {str(e)}")
            raise
            
    def comprehensive_impact_assessment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a comprehensive impact assessment using all models.
        
        Args:
            params (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, Any]: Comprehensive assessment results
        """
        logger.info(f"Performing comprehensive impact assessment for event: {params['event_type']} in {params['state']}")
        
        try:
            # Make predictions with all models
            property_damage = None
            casualty_risk = None
            severity = None
            
            try:
                property_damage = self.predict_property_damage(params)
            except Exception as e:
                logger.warning(f"Property damage prediction failed: {str(e)}")
                
            try:
                casualty_risk = self.predict_casualty_risk(params)
            except Exception as e:
                logger.warning(f"Casualty risk prediction failed: {str(e)}")
                
            try:
                severity = self.predict_severity(params)
            except Exception as e:
                logger.warning(f"Severity prediction failed: {str(e)}")
                
            # Compile the results and convert numpy types
            return numpy_to_python({
                "property_damage": property_damage,
                "casualty_risk": casualty_risk,
                "severity": severity,
                "event_params": params
            })
        except Exception as e:
            logger.error(f"Error in comprehensive impact assessment: {str(e)}")
            raise 