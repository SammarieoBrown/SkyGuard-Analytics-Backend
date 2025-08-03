import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib

from app.config import STATE_RISK_SCORES_PATH

logger = logging.getLogger(__name__)


class RegionalRiskModel:
    """
    Wrapper for regional risk scores model.
    Loads state risk scores and provides methods for risk assessment by region.
    """
    
    # Risk level categories and colors
    RISK_CATEGORIES = {
        "Low": "Risk level suggests minimal probability of severe weather events",
        "Moderate": "Risk level indicates occasional severe weather events with moderate impacts",
        "High": "Risk level shows frequent severe weather events with significant impacts",
        "Very High": "Risk level indicates high frequency of severe events with major impacts",
        "Extreme": "Risk level shows critical exposure to severe weather with catastrophic impacts"
    }
    
    RISK_COLORS = {
        "Low": "#b3cde3",       # Light blue
        "Moderate": "#8c96c6",  # Medium blue
        "High": "#8856a7",      # Purple
        "Very High": "#e66101", # Orange
        "Extreme": "#b2182b"    # Red
    }
    
    def __init__(self, risk_scores_path: Path = STATE_RISK_SCORES_PATH):
        """
        Initialize the model by loading risk scores from the specified path.
        
        Args:
            risk_scores_path (Path): Path to the pickle file containing risk scores
        """
        self.risk_data = self._load_risk_scores(risk_scores_path)
        self.risk_scores = None
        self.components_data = None
        
        # Extract the actual data structures
        if isinstance(self.risk_data, dict):
            if 'risk_scores' in self.risk_data:
                self.risk_scores = self.risk_data['risk_scores']
            if 'components' in self.risk_data and 'state_data' in self.risk_data['components']:
                self.components_data = self.risk_data['components']['state_data']
        
    def _load_risk_scores(self, risk_scores_path: Path):
        """
        Load risk scores from the pickle file.
        
        Args:
            risk_scores_path (Path): Path to the risk scores file
            
        Returns:
            The loaded risk scores data
        """
        try:
            # Try loading with joblib first (recommended for scikit-learn models)
            try:
                risk_scores = joblib.load(risk_scores_path)
                logger.info(f"Successfully loaded risk scores from {risk_scores_path} using joblib")
                return risk_scores
            except Exception as joblib_error:
                logger.warning(f"Failed to load risk scores with joblib: {str(joblib_error)}. Trying pickle...")
                
                # Fall back to pickle if joblib fails
                with open(risk_scores_path, 'rb') as file:
                    risk_scores = pickle.load(file)
                logger.info(f"Successfully loaded risk scores from {risk_scores_path} using pickle")
                return risk_scores
        except Exception as e:
            logger.error(f"Failed to load risk scores: {str(e)}")
            # Return empty DataFrame if file can't be loaded
            return pd.DataFrame()
    
    def get_state_risk(self, state_code: str) -> Dict[str, Any]:
        """
        Get risk assessment for a specific state.
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict[str, Any]: Risk assessment for the state
        """
        # Convert to uppercase for consistency
        state_code = state_code.upper()
        
        try:
            # Handle DC -> DI mapping (District of Columbia)
            lookup_code = 'DI' if state_code == 'DC' else state_code
            
            # If we have actual risk scores as a Series
            if isinstance(self.risk_scores, pd.Series) and lookup_code in self.risk_scores.index:
                risk_score = float(self.risk_scores[lookup_code])
                risk_category = self._get_risk_category(risk_score)
                
                # Get component data if available
                components = {
                    "frequency": 5.0,
                    "severity": 5.0,
                    "vulnerability": 5.0,
                    "trend": 1.0
                }
                
                historical_events = {
                    "total": 0,
                    "last_year": 0,
                    "average_annual": 0.0
                }
                
                # If we have components data, extract it
                if isinstance(self.components_data, pd.DataFrame) and lookup_code in self.components_data.index:
                    comp_data = self.components_data.loc[lookup_code]
                    historical_events = {
                        "total": int(comp_data.get('event_count', 0)),
                        "last_year": int(comp_data.get('event_count', 0) * 0.1),  # Estimate
                        "average_annual": float(comp_data.get('annual_events', 0))
                    }
                    
                    # Calculate component scores based on available data
                    if 'annual_events' in comp_data:
                        # Frequency score based on annual events (normalized to 0-10)
                        components['frequency'] = min(10, float(comp_data['annual_events']) / 100)
                    
                    if 'avg_damage' in comp_data:
                        # Severity score based on average damage (normalized)
                        components['severity'] = min(10, float(comp_data['avg_damage']) / 1e8)
                    
                    if 'population' in comp_data:
                        # Vulnerability based on population (normalized)
                        components['vulnerability'] = min(10, float(comp_data['population']) / 1e6)
                    
                    # Trend remains at 1.0 as we don't have trend data
                    components['trend'] = 1.0
                
                return {
                    "state_code": state_code,
                    "risk_score": risk_score,
                    "risk_category": risk_category,
                    "risk_description": self.RISK_CATEGORIES.get(risk_category, "Unknown risk level"),
                    "color_code": self.RISK_COLORS.get(risk_category, "#808080"),
                    "components": components,
                    "historical_events": historical_events
                }
            else:
                # If state not in data, return simulated data
                return self._get_simulated_state_risk(state_code)
                
        except Exception as e:
            logger.error(f"Error getting risk for state {state_code}: {str(e)}")
            return self._get_simulated_state_risk(state_code)
    
    def get_multi_state_risk(self, state_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get risk assessment for multiple states.
        
        Args:
            state_codes (List[str]): List of state codes
            
        Returns:
            Dict[str, Dict[str, Any]]: Risk assessments keyed by state code
        """
        results = {}
        for state in state_codes:
            results[state] = self.get_state_risk(state)
        return results
    
    def get_ranked_states(self, limit: int = 10, ascending: bool = False) -> List[Dict[str, Any]]:
        """
        Get states ranked by risk score.
        
        Args:
            limit (int): Number of states to return
            ascending (bool): Sort in ascending order if True, else descending
            
        Returns:
            List[Dict[str, Any]]: List of state risk assessments
        """
        try:
            if isinstance(self.risk_scores, pd.Series) and not self.risk_scores.empty:
                # Sort states by risk score
                sorted_scores = self.risk_scores.sort_values(ascending=ascending)
                
                # Filter to only valid 2-letter state codes and limit results
                valid_states = []
                for state_code, risk_score in sorted_scores.items():
                    if len(state_code) == 2:
                        state_result = self.get_state_risk(state_code)
                        # Add 'state' field for compatibility
                        state_result['state'] = state_code
                        valid_states.append(state_result)
                        if len(valid_states) >= limit:
                            break
                
                return valid_states
            else:
                # If no data, return simulated rankings
                return self._get_simulated_rankings(limit)
                
        except Exception as e:
            logger.error(f"Error getting ranked states: {str(e)}")
            return self._get_simulated_rankings(limit)
    
    def get_risk_by_event_type(self, event_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get risk assessment by event type across regions.
        
        Args:
            event_type (str): Type of weather event
            
        Returns:
            Dict[str, Dict[str, Any]]: Risk assessments keyed by state code
        """
        try:
            if isinstance(self.risk_scores, pd.DataFrame) and 'event_type_risks' in self.risk_scores.columns:
                results = {}
                
                for state_code in self.risk_scores.index:
                    state_data = self.risk_scores.loc[state_code]
                    event_risks = state_data.get('event_type_risks', {})
                    
                    if event_type in event_risks:
                        event_risk = event_risks[event_type]
                        risk_category = self._get_risk_category(event_risk)
                        
                        results[state_code] = {
                            "state_code": state_code,
                            "event_type": event_type,
                            "risk_score": float(event_risk),
                            "risk_category": risk_category,
                            "risk_description": self.RISK_CATEGORIES.get(risk_category, "Unknown risk level"),
                            "color_code": self.RISK_COLORS.get(risk_category, "#808080")
                        }
                
                return results
            else:
                # If no data or no event type risks, return simulated data
                return self._get_simulated_event_type_risks(event_type)
                
        except Exception as e:
            logger.error(f"Error getting risk by event type {event_type}: {str(e)}")
            return self._get_simulated_event_type_risks(event_type)
    
    def _get_risk_category(self, score: float) -> str:
        """
        Determine risk category based on risk score.
        
        Args:
            score (float): Risk score
            
        Returns:
            str: Risk category
        """
        if score < 3:
            return "Low"
        elif score < 5:
            return "Moderate"
        elif score < 7:
            return "High"
        elif score < 8.5:
            return "Very High"
        else:
            return "Extreme"
    
    def _get_simulated_state_risk(self, state_code: str) -> Dict[str, Any]:
        """
        Generate simulated risk assessment for a state when real data is not available.
        
        Args:
            state_code (str): State code
            
        Returns:
            Dict[str, Any]: Simulated risk assessment
        """
        # Deterministically generate a score based on state code to ensure consistency
        import hashlib
        # Create hash from state code
        hash_obj = hashlib.md5(state_code.encode())
        # Get a float between 0 and 10 based on the hash
        hash_int = int(hash_obj.hexdigest(), 16)
        risk_score = (hash_int % 1000) / 100  # 0.0 to 10.0
        
        risk_category = self._get_risk_category(risk_score)
        
        return {
            "state_code": state_code,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "risk_description": self.RISK_CATEGORIES.get(risk_category, "Unknown risk level"),
            "color_code": self.RISK_COLORS.get(risk_category, "#808080"),
            "components": {
                "frequency": (hash_int % 800) / 100,
                "severity": (hash_int % 900) / 100,
                "vulnerability": (hash_int % 700) / 100,
                "trend": (hash_int % 200) / 100 + 0.5
            },
            "historical_events": {
                "total": hash_int % 1000,
                "last_year": hash_int % 100,
                "average_annual": (hash_int % 500) / 10
            },
            "note": "Simulated data - actual risk scores not available"
        }
    
    def _get_simulated_rankings(self, limit: int) -> List[Dict[str, Any]]:
        """
        Generate simulated state rankings when real data is not available.
        
        Args:
            limit (int): Number of states to include
            
        Returns:
            List[Dict[str, Any]]: Simulated state rankings
        """
        # List of state codes for simulation
        state_codes = [
            "TX", "FL", "KS", "OK", "IA", 
            "MO", "AL", "MS", "AR", "LA",
            "NE", "SD", "ND", "IL", "IN", 
            "GA", "SC", "NC", "TN", "KY"
        ]
        
        # Limit to requested number
        state_codes = state_codes[:min(limit, len(state_codes))]
        
        return [self._get_simulated_state_risk(state) for state in state_codes]
    
    def _get_simulated_event_type_risks(self, event_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Generate simulated event type risks when real data is not available.
        
        Args:
            event_type (str): Event type
            
        Returns:
            Dict[str, Dict[str, Any]]: Simulated event type risks by state
        """
        # List of state codes for simulation
        state_codes = [
            "TX", "FL", "KS", "OK", "IA", 
            "MO", "AL", "MS", "AR", "LA"
        ]
        
        results = {}
        for state in state_codes:
            # Deterministically generate a score based on state code and event type
            import hashlib
            # Create hash from state code and event type
            hash_obj = hashlib.md5((state + event_type).encode())
            # Get a float between 0 and 10 based on the hash
            hash_int = int(hash_obj.hexdigest(), 16)
            risk_score = (hash_int % 1000) / 100  # 0.0 to 10.0
            
            risk_category = self._get_risk_category(risk_score)
            
            results[state] = {
                "state_code": state,
                "event_type": event_type,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "risk_description": self.RISK_CATEGORIES.get(risk_category, "Unknown risk level"),
                "color_code": self.RISK_COLORS.get(risk_category, "#808080"),
                "note": "Simulated data - actual risk scores not available"
            }
            
        return results 