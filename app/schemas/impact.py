from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PropertyDamagePredictionRequest(BaseModel):
    """Request schema for property damage prediction."""
    event_type: str = Field(..., description="Type of weather event")
    state: str = Field(..., description="State code (e.g., 'TX', 'CA')")
    magnitude: Optional[float] = Field(0.0, description="Event magnitude (e.g., wind speed)")
    duration_hours: Optional[float] = Field(1.0, description="Event duration in hours")
    month: Optional[int] = Field(6, description="Month of occurrence (1-12)")
    hour: Optional[int] = Field(12, description="Hour of occurrence (0-23)")
    latitude: Optional[float] = Field(None, description="Event latitude")
    longitude: Optional[float] = Field(None, description="Event longitude")
    tor_f_scale: Optional[str] = Field(None, description="Tornado F/EF scale if applicable")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "Thunderstorm Wind",
                "state": "TX",
                "magnitude": 65.0,
                "duration_hours": 1.5,
                "month": 6,
                "hour": 14,
                "latitude": 32.7767,
                "longitude": -96.7970,
                "tor_f_scale": None
            }
        }
    }


class PropertyDamagePredictionResponse(BaseModel):
    """Response schema for property damage prediction."""
    predicted_damage: float = Field(..., description="Predicted property damage in USD")
    prediction_range: Dict[str, float] = Field(
        ..., 
        description="Range of potential damage (low, medium, high estimates)"
    )
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    influential_factors: List[Dict[str, Any]] = Field(
        ..., 
        description="Factors that influenced the prediction"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_damage": 75000.0,
                "prediction_range": {
                    "low_estimate": 50000.0,
                    "expected": 75000.0,
                    "high_estimate": 100000.0
                },
                "confidence_score": 0.85,
                "influential_factors": [
                    {"factor": "event_type", "importance": 0.35},
                    {"factor": "magnitude", "importance": 0.25},
                    {"factor": "state", "importance": 0.20},
                    {"factor": "duration_hours", "importance": 0.15}
                ]
            }
        }
    }


class CasualtyRiskPredictionRequest(BaseModel):
    """Request schema for casualty risk prediction."""
    event_type: str = Field(..., description="Type of weather event")
    state: str = Field(..., description="State code (e.g., 'TX', 'CA')")
    magnitude: Optional[float] = Field(0.0, description="Event magnitude (e.g., wind speed)")
    duration_hours: Optional[float] = Field(1.0, description="Event duration in hours")
    month: Optional[int] = Field(6, description="Month of occurrence (1-12)")
    hour: Optional[int] = Field(12, description="Hour of occurrence (0-23)")
    tor_f_scale: Optional[str] = Field(None, description="Tornado F/EF scale if applicable")
    cz_name: Optional[str] = Field(None, description="County/zone name")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "Tornado",
                "state": "OK",
                "magnitude": 150.0,
                "duration_hours": 0.5,
                "month": 5,
                "hour": 16,
                "tor_f_scale": "EF3",
                "cz_name": "Oklahoma County"
            }
        }
    }


class CasualtyRiskPredictionResponse(BaseModel):
    """Response schema for casualty risk prediction."""
    casualty_risk_score: float = Field(..., description="Raw casualty risk score (0-1)")
    risk_level: float = Field(..., description="Standardized risk level (0-10)")
    risk_category: str = Field(..., description="Risk category (Low, Moderate, High, Very High)")
    probability: Dict[str, float] = Field(..., description="Probability of casualties")
    population_risk_factors: Dict[str, float] = Field(
        ..., 
        description="Population-related risk factors"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "casualty_risk_score": 0.72,
                "risk_level": 7.2,
                "risk_category": "Very High",
                "probability": {
                    "no_casualties": 0.28,
                    "casualties": 0.72
                },
                "population_risk_factors": {
                    "population_density": 0.7,
                    "vulnerable_population": 0.5,
                    "infrastructure_resilience": 0.6
                }
            }
        }
    }


class SeverityPredictionRequest(BaseModel):
    """Request schema for severity level prediction."""
    event_type: str = Field(..., description="Type of weather event")
    state: str = Field(..., description="State code (e.g., 'TX', 'CA')")
    magnitude: Optional[float] = Field(0.0, description="Event magnitude (e.g., wind speed)")
    duration_hours: Optional[float] = Field(1.0, description="Event duration in hours")
    property_damage: Optional[float] = Field(0.0, description="Property damage in USD")
    crop_damage: Optional[float] = Field(0.0, description="Crop damage in USD")
    injuries: Optional[int] = Field(0, description="Number of injuries")
    deaths: Optional[int] = Field(0, description="Number of deaths")
    month: Optional[int] = Field(6, description="Month of occurrence (1-12)")
    hour: Optional[int] = Field(12, description="Hour of occurrence (0-23)")
    tor_f_scale: Optional[str] = Field(None, description="Tornado F/EF scale if applicable")
    cz_type: Optional[str] = Field("Z", description="County/zone type (C=County, Z=Zone)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "Tornado",
                "state": "OK",
                "magnitude": 150.0,
                "duration_hours": 0.5,
                "property_damage": 500000.0,
                "crop_damage": 50000.0,
                "injuries": 15,
                "deaths": 2,
                "month": 5,
                "hour": 16,
                "tor_f_scale": "EF3",
                "cz_type": "C"
            }
        }
    }


class SeverityPredictionResponse(BaseModel):
    """Response schema for severity level prediction."""
    severity_class: str = Field(..., description="Severity classification")
    description: str = Field(..., description="Detailed description of the severity level")
    color_code: str = Field(..., description="Color code for visualization")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    class_probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability distribution across severity classes"
    )
    impact_factors: Dict[str, Any] = Field(
        ..., 
        description="Factors contributing to the severity assessment"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "severity_class": "Severe",
                "description": "Widespread damage across region. Many casualties expected, extensive property damage.",
                "color_code": "#d6604d",
                "confidence_score": 0.85,
                "class_probabilities": {
                    "Minor": 0.05,
                    "Moderate": 0.10,
                    "Significant": 0.15,
                    "Severe": 0.60,
                    "Catastrophic": 0.10
                },
                "impact_factors": {
                    "property_damage": {
                        "value": 500000.0,
                        "impact_level": "High"
                    },
                    "crop_damage": {
                        "value": 50000.0,
                        "impact_level": "Moderate"
                    },
                    "casualties": {
                        "injuries": 15,
                        "deaths": 2,
                        "total": 17,
                        "impact_level": "Moderate"
                    },
                    "tornado_scale": {
                        "value": "EF3",
                        "impact_level": "High"
                    }
                }
            }
        }
    } 