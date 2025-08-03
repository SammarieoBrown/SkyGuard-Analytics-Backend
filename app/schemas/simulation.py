from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from enum import Enum


class EventParameters(BaseModel):
    """Base event parameters."""
    event_type: str = Field(..., description="Type of weather event")
    state: str = Field(..., description="State code")
    magnitude: Optional[float] = Field(0.0, description="Event magnitude")
    duration_hours: Optional[float] = Field(1.0, description="Event duration in hours")
    property_damage: Optional[float] = Field(0.0, description="Property damage in USD")
    crop_damage: Optional[float] = Field(0.0, description="Crop damage in USD")
    injuries: Optional[int] = Field(0, description="Number of injuries")
    deaths: Optional[int] = Field(0, description="Number of deaths")
    month: Optional[int] = Field(6, ge=1, le=12, description="Month (1-12)")
    hour: Optional[int] = Field(12, ge=0, le=23, description="Hour (0-23)")
    latitude: Optional[float] = Field(None, description="Event latitude")
    longitude: Optional[float] = Field(None, description="Event longitude")
    tor_f_scale: Optional[str] = Field(None, description="Tornado F/EF scale")
    cz_type: Optional[str] = Field("Z", description="County/zone type (C=County, Z=Zone)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "Tornado",
                "state": "OK",
                "magnitude": 150.0,
                "duration_hours": 0.5,
                "property_damage": 0.0,
                "crop_damage": 0.0,
                "injuries": 0,
                "deaths": 0,
                "month": 5,
                "hour": 16,
                "latitude": 35.4823,
                "longitude": -97.7350,
                "tor_f_scale": "EF3",
                "cz_type": "C"
            }
        }
    }


class ModificationType(str, Enum):
    SET = "set"
    ADD = "add"
    MULTIPLY = "multiply"


class ParameterModification(BaseModel):
    """Parameter modification specification."""
    parameter: str = Field(..., description="Parameter name to modify")
    modification_type: ModificationType = Field(..., description="Type of modification")
    value: Union[float, int, str] = Field(..., description="New value or modification factor")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter": "magnitude",
                "modification_type": "multiply",
                "value": 1.5
            }
        }
    }


class ScenarioRequest(BaseModel):
    """Scenario simulation request."""
    base_event: EventParameters = Field(..., description="Base event parameters")
    modifications: List[ParameterModification] = Field(..., description="Parameter modifications")
    include_uncertainty: Optional[bool] = Field(True, description="Include uncertainty metrics in the response")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "base_event": {
                    "event_type": "Tornado",
                    "state": "OK",
                    "magnitude": 150.0,
                    "duration_hours": 0.5,
                    "month": 5,
                    "hour": 16,
                    "tor_f_scale": "EF3"
                },
                "modifications": [
                    {
                        "parameter": "magnitude",
                        "modification_type": "multiply",
                        "value": 1.5
                    },
                    {
                        "parameter": "tor_f_scale",
                        "modification_type": "set",
                        "value": "EF4"
                    }
                ],
                "include_uncertainty": True
            }
        }
    }


class BatchScenarioRequest(BaseModel):
    """Batch scenario simulation request."""
    base_event: EventParameters = Field(..., description="Base event parameters")
    scenario_sets: List[List[ParameterModification]] = Field(..., description="List of modification sets")
    include_uncertainty: Optional[bool] = Field(True, description="Include uncertainty metrics in the response")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "base_event": {
                    "event_type": "Tornado",
                    "state": "OK",
                    "magnitude": 150.0,
                    "duration_hours": 0.5,
                    "month": 5,
                    "hour": 16,
                    "tor_f_scale": "EF3"
                },
                "scenario_sets": [
                    [
                        {
                            "parameter": "magnitude",
                            "modification_type": "multiply",
                            "value": 1.5
                        }
                    ],
                    [
                        {
                            "parameter": "tor_f_scale",
                            "modification_type": "set",
                            "value": "EF4"
                        }
                    ],
                    [
                        {
                            "parameter": "magnitude",
                            "modification_type": "multiply",
                            "value": 1.5
                        },
                        {
                            "parameter": "tor_f_scale",
                            "modification_type": "set",
                            "value": "EF4"
                        }
                    ]
                ],
                "include_uncertainty": True
            }
        }
    }


class SensitivityRequest(BaseModel):
    """Sensitivity analysis request."""
    base_event: EventParameters = Field(..., description="Base event parameters")
    parameters: List[str] = Field(..., description="Parameters to analyze")
    variation_range: Optional[float] = Field(0.5, ge=0.1, le=2.0, description="Variation range (0.1-2.0)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "base_event": {
                    "event_type": "Tornado",
                    "state": "OK",
                    "magnitude": 150.0,
                    "duration_hours": 0.5,
                    "month": 5,
                    "hour": 16,
                    "tor_f_scale": "EF3"
                },
                "parameters": ["magnitude", "duration_hours"],
                "variation_range": 0.5
            }
        }
    }


class PredictionResult(BaseModel):
    """Base prediction result."""
    property_damage: Optional[Dict[str, Any]] = Field(None, description="Property damage prediction")
    casualty_risk: Optional[Dict[str, Any]] = Field(None, description="Casualty risk prediction")
    severity: Optional[Dict[str, Any]] = Field(None, description="Severity prediction")


class ConfidenceInterval(BaseModel):
    """Confidence interval for a prediction."""
    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")


class ScenarioResult(BaseModel):
    """Scenario simulation result."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    base_prediction: PredictionResult = Field(..., description="Base event prediction")
    modified_prediction: PredictionResult = Field(..., description="Modified event prediction")
    parameter_changes: Dict[str, Any] = Field(..., description="Summary of parameter changes")
    impact_analysis: Dict[str, Any] = Field(..., description="Analysis of changes in predictions")
    confidence_intervals: Optional[Dict[str, ConfidenceInterval]] = Field(None, description="Confidence intervals for predictions")
    uncertainty_metrics: Optional[Dict[str, float]] = Field(None, description="Uncertainty metrics for predictions")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "scenario_id": "scenario_123456",
                "base_prediction": {
                    "property_damage": {
                        "predicted_damage": 75000.0,
                        "prediction_range": {
                            "low_estimate": 50000.0,
                            "expected": 75000.0,
                            "high_estimate": 100000.0
                        }
                    },
                    "casualty_risk": {
                        "casualty_risk_score": 0.35,
                        "risk_level": 3.5,
                        "risk_category": "Moderate"
                    },
                    "severity": {
                        "severity_class": "Significant",
                        "confidence_score": 0.75
                    }
                },
                "modified_prediction": {
                    "property_damage": {
                        "predicted_damage": 150000.0,
                        "prediction_range": {
                            "low_estimate": 100000.0,
                            "expected": 150000.0,
                            "high_estimate": 200000.0
                        }
                    },
                    "casualty_risk": {
                        "casualty_risk_score": 0.65,
                        "risk_level": 6.5,
                        "risk_category": "High"
                    },
                    "severity": {
                        "severity_class": "Severe",
                        "confidence_score": 0.80
                    }
                },
                "parameter_changes": {
                    "magnitude": {
                        "original": 150.0,
                        "modified": 225.0,
                        "change_factor": 1.5,
                        "change_type": "multiply"
                    },
                    "tor_f_scale": {
                        "original": "EF3",
                        "modified": "EF4",
                        "change_type": "set"
                    }
                },
                "impact_analysis": {
                    "property_damage_change": {
                        "change_amount": 75000.0,
                        "change_percent": 100.0,
                        "significance": "High"
                    },
                    "casualty_risk_change": {
                        "change_amount": 0.3,
                        "change_percent": 85.7,
                        "category_change": True,
                        "significance": "High"
                    },
                    "severity_change": {
                        "category_change": True,
                        "significance": "High"
                    },
                    "overall_impact": "High"
                },
                "confidence_intervals": {
                    "property_damage": {
                        "lower": 120000.0,
                        "upper": 180000.0
                    },
                    "casualty_risk": {
                        "lower": 0.55,
                        "upper": 0.75
                    }
                },
                "uncertainty_metrics": {
                    "property_damage_cv": 0.2,
                    "casualty_risk_cv": 0.15
                }
            }
        }
    }


class BatchScenarioResponse(BaseModel):
    """Batch scenario simulation response."""
    batch_id: str = Field(..., description="Batch identifier")
    scenarios: List[ScenarioResult] = Field(..., description="List of scenario results")
    summary: Dict[str, Any] = Field(..., description="Summary of batch results")


class SensitivityAnalysisResult(BaseModel):
    """Sensitivity analysis result."""
    analysis_id: str = Field(..., description="Analysis identifier")
    base_prediction: PredictionResult = Field(..., description="Base prediction")
    parameter_sensitivities: Dict[str, Dict[str, Any]] = Field(..., description="Parameter sensitivities")
    visualization_data: Dict[str, Any] = Field(..., description="Data for visualization")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis_id": "sensitivity_123456",
                "base_prediction": {
                    "property_damage": {
                        "predicted_damage": 75000.0
                    },
                    "casualty_risk": {
                        "casualty_risk_score": 0.35
                    },
                    "severity": {
                        "severity_class": "Significant"
                    }
                },
                "parameter_sensitivities": {
                    "magnitude": {
                        "property_damage_elasticity": 1.2,
                        "casualty_risk_elasticity": 0.8,
                        "overall_importance": 0.65
                    },
                    "duration_hours": {
                        "property_damage_elasticity": 0.5,
                        "casualty_risk_elasticity": 0.3,
                        "overall_importance": 0.35
                    }
                },
                "visualization_data": {
                    "parameter_importance": [
                        {
                            "parameter": "magnitude",
                            "importance": 0.65
                        },
                        {
                            "parameter": "duration_hours",
                            "importance": 0.35
                        }
                    ],
                    "elasticity_matrix": [
                        {
                            "parameter": "magnitude",
                            "property_damage": 1.2,
                            "casualty_risk": 0.8
                        },
                        {
                            "parameter": "duration_hours",
                            "property_damage": 0.5,
                            "casualty_risk": 0.3
                        }
                    ]
                }
            }
        }
    } 