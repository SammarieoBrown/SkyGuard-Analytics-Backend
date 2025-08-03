from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class StateRiskRequest(BaseModel):
    """Request schema for state risk assessment."""
    state_code: str = Field(..., description="Two-letter state code (e.g., 'TX', 'CA')")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "state_code": "TX"
            }
        }
    }


class RiskComponentsResponse(BaseModel):
    """Response schema for risk components."""
    frequency: float = Field(..., description="Frequency score based on historical event frequency")
    severity: float = Field(..., description="Severity score based on historical impact")
    vulnerability: float = Field(..., description="Vulnerability score based on population factors")
    trend: float = Field(..., description="Trend factor indicating increasing or decreasing risk")


class HistoricalEventsResponse(BaseModel):
    """Response schema for historical event statistics."""
    total: int = Field(..., description="Total number of historical events")
    last_year: int = Field(..., description="Number of events in the last year")
    average_annual: float = Field(..., description="Average annual events")


class StateRiskResponse(BaseModel):
    """Response schema for state risk assessment."""
    state_code: str = Field(..., description="Two-letter state code")
    risk_score: float = Field(..., description="Composite risk score (0-10)")
    risk_category: str = Field(..., description="Risk category (Low, Moderate, High, Very High, Extreme)")
    risk_description: str = Field(..., description="Description of the risk level")
    color_code: str = Field(..., description="Color hex code for visualization")
    components: RiskComponentsResponse = Field(..., description="Risk score components")
    historical_events: HistoricalEventsResponse = Field(..., description="Historical event statistics")
    note: Optional[str] = Field(None, description="Additional notes or data source information")


class MultiStateRiskRequest(BaseModel):
    """Request schema for multi-state risk assessment."""
    state_codes: List[str] = Field(..., description="List of two-letter state codes")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "state_codes": ["TX", "FL", "CA"]
            }
        }
    }


class MultiStateRiskResponse(BaseModel):
    """Response schema for multi-state risk assessment."""
    risks: Dict[str, StateRiskResponse] = Field(..., description="Risk assessments by state code")


class RankingRequest(BaseModel):
    """Request schema for state risk rankings."""
    limit: Optional[int] = Field(10, description="Number of states to include")
    ascending: Optional[bool] = Field(False, description="Sort in ascending order if True")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "limit": 10,
                "ascending": False
            }
        }
    }


class EventTypeRiskRequest(BaseModel):
    """Request schema for event type risk assessment."""
    event_type: str = Field(..., description="Type of weather event")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "Tornado"
            }
        }
    }


class EventTypeRiskResponse(BaseModel):
    """Response schema for state risk by event type."""
    state_code: str = Field(..., description="Two-letter state code")
    event_type: str = Field(..., description="Type of weather event")
    risk_score: float = Field(..., description="Risk score for this event type (0-10)")
    risk_category: str = Field(..., description="Risk category")
    risk_description: str = Field(..., description="Description of the risk level")
    color_code: str = Field(..., description="Color hex code for visualization")
    note: Optional[str] = Field(None, description="Additional notes or data source information")


class EventTypeRisksResponse(BaseModel):
    """Response schema for event type risks across states."""
    risks: Dict[str, EventTypeRiskResponse] = Field(..., description="Risk assessments by state code") 