from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List

from app.schemas.risk import (
    StateRiskRequest,
    StateRiskResponse,
    MultiStateRiskRequest,
    MultiStateRiskResponse,
    RankingRequest,
    EventTypeRiskRequest,
    EventTypeRisksResponse
)
from app.core.services.risk_service import RegionalRiskService

router = APIRouter()


def get_risk_service():
    """Dependency to get the regional risk service."""
    return RegionalRiskService()


@router.post("/state", response_model=StateRiskResponse)
async def get_state_risk(
    request: StateRiskRequest,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get risk assessment for a specific state.
    
    This endpoint provides risk scores, categories, and components for a requested state.
    """
    try:
        result = service.get_state_risk(request.state_code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment error: {str(e)}")


@router.get("/state/{state_code}", response_model=StateRiskResponse)
async def get_state_risk_by_path(
    state_code: str,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get risk assessment for a specific state using path parameter.
    
    This endpoint provides risk scores, categories, and components for a requested state.
    """
    try:
        result = service.get_state_risk(state_code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment error: {str(e)}")


@router.post("/multi-state", response_model=MultiStateRiskResponse)
async def get_multi_state_risk(
    request: MultiStateRiskRequest,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get risk assessment for multiple states.
    
    This endpoint provides risk assessments for multiple states in one request.
    """
    try:
        result = service.get_multi_state_risk(request.state_codes)
        return {"risks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment error: {str(e)}")


@router.post("/rankings", response_model=List[StateRiskResponse])
async def get_ranked_states(
    request: RankingRequest,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get states ranked by risk score.
    
    This endpoint provides a list of states ranked by their risk scores.
    """
    try:
        result = service.get_ranked_states(
            limit=request.limit, 
            ascending=request.ascending
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk ranking error: {str(e)}")


@router.get("/rankings", response_model=List[StateRiskResponse])
async def get_ranked_states_default(
    limit: int = 10,
    ascending: bool = False,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get states ranked by risk score using query parameters.
    
    This endpoint provides a list of states ranked by their risk scores.
    """
    try:
        result = service.get_ranked_states(
            limit=limit, 
            ascending=ascending
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk ranking error: {str(e)}")


@router.post("/event-type", response_model=EventTypeRisksResponse)
async def get_risk_by_event_type(
    request: EventTypeRiskRequest,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get risk assessment by event type across regions.
    
    This endpoint provides risk assessments for a specific event type across different states.
    """
    try:
        result = service.get_risk_by_event_type(request.event_type)
        return {"risks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event type risk assessment error: {str(e)}")


@router.get("/event-type/{event_type}", response_model=EventTypeRisksResponse)
async def get_risk_by_event_type_path(
    event_type: str,
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get risk assessment by event type across regions using path parameter.
    
    This endpoint provides risk assessments for a specific event type across different states.
    """
    try:
        result = service.get_risk_by_event_type(event_type)
        return {"risks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Event type risk assessment error: {str(e)}")


@router.get("/summary")
async def get_risk_summary(
    service: RegionalRiskService = Depends(get_risk_service)
):
    """
    Get a summary of risk across all regions.
    
    This endpoint provides summary statistics of risk across all regions.
    """
    try:
        return service.get_risk_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk summary error: {str(e)}")


@router.get("/healthcheck")
async def healthcheck():
    """Health check endpoint for the regional risk API."""
    return {"status": "ok", "service": "regional-risk"} 