from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from app.schemas.impact import (
    PropertyDamagePredictionRequest, 
    PropertyDamagePredictionResponse,
    CasualtyRiskPredictionRequest,
    CasualtyRiskPredictionResponse,
    SeverityPredictionRequest,
    SeverityPredictionResponse
)
from app.core.services.impact_service import ImpactForecastingService

router = APIRouter()

# Create a singleton instance of the service
# This ensures we reuse the same service instance with preloaded models
_impact_service_instance = None

def get_impact_service():
    """Dependency to get the impact forecasting service (singleton pattern)."""
    global _impact_service_instance
    if _impact_service_instance is None:
        _impact_service_instance = ImpactForecastingService()
    return _impact_service_instance


@router.post("/property-damage", response_model=PropertyDamagePredictionResponse)
async def predict_property_damage(
    request: PropertyDamagePredictionRequest,
    service: ImpactForecastingService = Depends(get_impact_service)
):
    """
    Predict property damage for a weather event.
    
    This endpoint takes weather event parameters and returns predicted property damage
    along with confidence levels and influential factors.
    """
    try:
        result = service.predict_property_damage(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/casualty-risk", response_model=CasualtyRiskPredictionResponse)
async def predict_casualty_risk(
    request: CasualtyRiskPredictionRequest,
    service: ImpactForecastingService = Depends(get_impact_service)
):
    """
    Predict casualty risk for a weather event.
    
    This endpoint takes weather event parameters and returns casualty risk assessment
    including risk scores, categories, and population risk factors.
    """
    try:
        result = service.predict_casualty_risk(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/severity", response_model=SeverityPredictionResponse)
async def predict_severity(
    request: SeverityPredictionRequest,
    service: ImpactForecastingService = Depends(get_impact_service)
):
    """
    Predict severity level for a weather event.
    
    This endpoint takes weather event parameters and returns a severity assessment
    including severity class, description, and impact factors.
    """
    try:
        result = service.predict_severity(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/comprehensive-assessment")
async def comprehensive_assessment(
    request: SeverityPredictionRequest,  # Using the most complete schema
    service: ImpactForecastingService = Depends(get_impact_service)
):
    """
    Perform a comprehensive impact assessment using all models.
    
    This endpoint takes weather event parameters and returns a comprehensive assessment
    including property damage, casualty risk, and severity predictions.
    """
    try:
        result = service.comprehensive_impact_assessment(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment error: {str(e)}")


@router.get("/healthcheck")
async def healthcheck():
    """Health check endpoint for the impact forecasting API."""
    return {"status": "ok", "service": "impact-forecasting"} 