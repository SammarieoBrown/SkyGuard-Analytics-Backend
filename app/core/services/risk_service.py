from typing import Dict, Any, List, Optional
import logging
from app.core.models.model_manager import model_manager
from app.utils.json_encoder import numpy_to_python

logger = logging.getLogger(__name__)


class RegionalRiskService:
    """
    Service for regional risk assessment.
    This service acts as an intermediary between the API and the underlying risk model.
    """
    
    def __init__(self):
        """Initialize service with required models from model manager."""
        self.risk_model = model_manager.get_risk_model()
    
    def get_state_risk(self, state_code: str) -> Dict[str, Any]:
        """
        Get risk assessment for a specific state.
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict[str, Any]: Risk assessment for the state
        """
        logger.info(f"Getting risk assessment for state: {state_code}")
        
        try:
            result = self.risk_model.get_state_risk(state_code)
            return numpy_to_python(result)
        except Exception as e:
            logger.error(f"Error in state risk assessment service: {str(e)}")
            raise
    
    def get_multi_state_risk(self, state_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get risk assessment for multiple states.
        
        Args:
            state_codes (List[str]): List of state codes
            
        Returns:
            Dict[str, Dict[str, Any]]: Risk assessments keyed by state code
        """
        logger.info(f"Getting risk assessment for {len(state_codes)} states")
        
        try:
            result = self.risk_model.get_multi_state_risk(state_codes)
            return numpy_to_python(result)
        except Exception as e:
            logger.error(f"Error in multi-state risk assessment service: {str(e)}")
            raise
    
    def get_ranked_states(self, limit: int = 10, ascending: bool = False) -> List[Dict[str, Any]]:
        """
        Get states ranked by risk score.
        
        Args:
            limit (int): Number of states to return
            ascending (bool): Sort in ascending order if True, else descending
            
        Returns:
            List[Dict[str, Any]]: List of state risk assessments
        """
        logger.info(f"Getting top {limit} states ranked by {'lowest' if ascending else 'highest'} risk")
        
        try:
            result = self.risk_model.get_ranked_states(limit=limit, ascending=ascending)
            return numpy_to_python(result)
        except Exception as e:
            logger.error(f"Error in ranked states service: {str(e)}")
            raise
    
    def get_risk_by_event_type(self, event_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get risk assessment by event type across regions.
        
        Args:
            event_type (str): Type of weather event
            
        Returns:
            Dict[str, Dict[str, Any]]: Risk assessments keyed by state code
        """
        logger.info(f"Getting risk assessment for event type: {event_type}")
        
        try:
            result = self.risk_model.get_risk_by_event_type(event_type)
            return numpy_to_python(result)
        except Exception as e:
            logger.error(f"Error in event type risk assessment service: {str(e)}")
            raise
            
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get a summary of risk across all regions.
        
        Returns:
            Dict[str, Any]: Summary statistics of risk
        """
        logger.info(f"Generating risk summary across all regions")
        
        try:
            # Get top states by risk
            top_states = self.risk_model.get_ranked_states(limit=5)
            
            # Get count by risk category
            risk_categories = {}
            for state in top_states:
                category = state['risk_category']
                risk_categories[category] = risk_categories.get(category, 0) + 1
                
            # Create summary
            return numpy_to_python({
                "highest_risk_states": [state['state_code'] for state in top_states[:3]],
                "highest_risk_score": max([state['risk_score'] for state in top_states]) if top_states else 0,
                "risk_categories": risk_categories
            })
        except Exception as e:
            logger.error(f"Error generating risk summary: {str(e)}")
            raise 