"""
Model Manager - Singleton pattern for model loading.
Prevents multiple loads of the same model.
"""
import logging
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for ML models to prevent multiple loads."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._models = {}
        logger.info("ModelManager initialized")
    
    def get_property_damage_model(self):
        """Get or create property damage model."""
        if 'property_damage' not in self._models:
            logger.info("Loading property damage model...")
            from app.core.models.damage_model import PropertyDamageModel
            self._models['property_damage'] = PropertyDamageModel()
        return self._models['property_damage']
    
    def get_casualty_risk_model(self):
        """Get or create casualty risk model."""
        if 'casualty_risk' not in self._models:
            logger.info("Loading casualty risk model...")
            from app.core.models.casualty_model import CasualtyRiskModel
            self._models['casualty_risk'] = CasualtyRiskModel()
        return self._models['casualty_risk']
    
    def get_severity_model(self):
        """Get or create severity model."""
        if 'severity' not in self._models:
            logger.info("Loading severity model...")
            from app.core.models.severity_model import SeverityModel
            self._models['severity'] = SeverityModel()
        return self._models['severity']
    
    def get_risk_model(self):
        """Get or create risk model."""
        if 'risk' not in self._models:
            logger.info("Loading risk model...")
            from app.core.models.risk_model import RegionalRiskModel
            self._models['risk'] = RegionalRiskModel()
        return self._models['risk']
    
    def get_weather_nowcasting_model(self):
        """Get or create weather nowcasting model."""
        if 'weather_nowcasting' not in self._models:
            logger.info("Loading weather nowcasting model...")
            from app.core.models.weather_nowcasting_model import WeatherNowcastingModel
            model = WeatherNowcastingModel()
            # Load the model immediately to ensure it's ready
            if model.load_model():
                self._models['weather_nowcasting'] = model
                logger.info("Weather nowcasting model loaded and cached successfully")
            else:
                logger.error("Failed to load weather nowcasting model")
                raise RuntimeError("Weather nowcasting model failed to load")
        return self._models['weather_nowcasting']
    
    def get_model_health_status(self):
        """Get health status of all models."""
        health_status = {}
        
        for model_name, model in self._models.items():
            if hasattr(model, 'health_check'):
                health_status[model_name] = model.health_check()
            else:
                health_status[model_name] = {"status": "unknown", "health_check_not_implemented": True}
        
        return health_status
    
    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()
        logger.info("Model cache cleared")


# Global instance
model_manager = ModelManager()