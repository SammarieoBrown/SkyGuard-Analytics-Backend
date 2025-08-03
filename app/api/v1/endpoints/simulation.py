from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from typing import Dict, Any, List, Optional

from app.schemas.simulation import (
    ScenarioRequest,
    BatchScenarioRequest,
    SensitivityRequest,
    ScenarioResult,
    BatchScenarioResponse,
    SensitivityAnalysisResult
)
from app.core.services.simulation_service import ScenarioSimulationService
from app.utils import numpy_to_python

router = APIRouter()


def get_simulation_service():
    """Dependency to get the simulation service."""
    return ScenarioSimulationService()


@router.post("/scenario", response_model=ScenarioResult)
async def simulate_scenario(
    request: ScenarioRequest,
    service: ScenarioSimulationService = Depends(get_simulation_service)
):
    """
    Simulate a scenario by modifying event parameters.
    
    This endpoint takes a base event and parameter modifications, simulates the modified scenario,
    and returns the simulation results including impact analysis.
    """
    try:
        result = service.simulate_scenario(
            base_event=request.base_event.dict(),
            modifications=[mod.dict() for mod in request.modifications],
            include_uncertainty=request.include_uncertainty
        )
        # Convert numpy types before returning
        return numpy_to_python(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.post("/batch", response_model=BatchScenarioResponse)
async def batch_simulate(
    request: BatchScenarioRequest,
    service: ScenarioSimulationService = Depends(get_simulation_service)
):
    """
    Run batch scenario simulations.
    
    This endpoint takes a base event and multiple sets of parameter modifications,
    simulates each scenario, and returns the aggregated results.
    """
    try:
        # Convert each scenario set to a list of dictionaries
        scenario_sets = [[mod.dict() for mod in scenario_set] for scenario_set in request.scenario_sets]
        
        result = service.batch_simulate(
            base_event=request.base_event.dict(),
            scenario_sets=scenario_sets,
            include_uncertainty=request.include_uncertainty
        )
        # Convert numpy types before returning
        return numpy_to_python(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch simulation error: {str(e)}")


@router.post("/sensitivity", response_model=SensitivityAnalysisResult)
async def analyze_sensitivity(
    request: SensitivityRequest,
    service: ScenarioSimulationService = Depends(get_simulation_service)
):
    """
    Perform sensitivity analysis on selected parameters.
    
    This endpoint analyzes how changes to selected parameters affect predictions,
    and returns the parameter sensitivities and visualization data.
    """
    try:
        result = service.perform_sensitivity_analysis(
            base_event=request.base_event.dict(),
            parameters=request.parameters,
            variation_range=request.variation_range
        )
        # Convert numpy types before returning
        return numpy_to_python(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis error: {str(e)}")


@router.get("/scenario/{scenario_id}")
async def get_scenario(
    scenario_id: str = Path(..., description="Unique scenario identifier"),
    format: str = Query("json", description="Response format (json or report)"),
    service: ScenarioSimulationService = Depends(get_simulation_service)
):
    """
    Get a saved scenario by ID.
    
    This endpoint retrieves a previously run scenario by its ID.
    The format parameter determines the response format (JSON or human-readable report).
    """
    try:
        scenario = service.get_saved_scenario(scenario_id)
        
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        
        if format.lower() == "report":
            # Convert to human-readable report format
            return {
                "scenario_id": scenario.get("scenario_id"),
                "timestamp": scenario.get("timestamp"),
                "report": _generate_scenario_report(scenario),
                "format": "text"
            }
        
        # Default format is JSON
        return scenario
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving scenario: {str(e)}")


@router.get("/healthcheck")
async def healthcheck():
    """Health check endpoint for the simulation API."""
    return {"status": "ok", "service": "scenario-simulation"}


def _generate_scenario_report(scenario: Dict[str, Any]) -> str:
    """
    Generate a human-readable report for a scenario.
    
    Args:
        scenario (Dict[str, Any]): Scenario result
        
    Returns:
        str: Formatted report
    """
    # Basic scenario info
    report = [
        f"SCENARIO REPORT: {scenario.get('scenario_id', 'Unknown')}",
        f"Timestamp: {scenario.get('timestamp', 'Unknown')}",
        "\n### BASE EVENT ###"
    ]
    
    # Base event info
    base_event = scenario.get('base_event', {})
    report.append(f"Event Type: {base_event.get('event_type', 'Unknown')}")
    report.append(f"State: {base_event.get('state', 'Unknown')}")
    
    for key, value in base_event.items():
        if key not in ['event_type', 'state']:
            report.append(f"{key}: {value}")
    
    # Modifications
    report.append("\n### MODIFICATIONS ###")
    for mod in scenario.get('modifications', []):
        param = mod.get('parameter', 'Unknown')
        mod_type = mod.get('modification_type', 'Unknown')
        value = mod.get('value', 'Unknown')
        report.append(f"{param}: {mod_type} to {value}")
    
    # Impact summary
    report.append("\n### IMPACT SUMMARY ###")
    impact = scenario.get('impact_analysis', {})
    report.append(f"Overall Impact: {impact.get('overall_impact', 'Unknown')}")
    
    # Predictions
    report.append("\n### PREDICTION CHANGES ###")
    base_pred = scenario.get('base_prediction', {})
    mod_pred = scenario.get('modified_prediction', {})
    
    # Property damage
    base_damage = base_pred.get('property_damage', {}).get('predicted_damage', 0)
    mod_damage = mod_pred.get('property_damage', {}).get('predicted_damage', 0)
    report.append(f"Property Damage: ${base_damage:,.2f} → ${mod_damage:,.2f}")
    
    # Casualty risk
    base_risk = base_pred.get('casualty_risk', {}).get('casualty_risk_score', 0)
    mod_risk = mod_pred.get('casualty_risk', {}).get('casualty_risk_score', 0)
    report.append(f"Casualty Risk: {base_risk:.2f} → {mod_risk:.2f}")
    
    # Severity
    base_severity = base_pred.get('severity', {}).get('severity_class', 'Unknown')
    mod_severity = mod_pred.get('severity', {}).get('severity_class', 'Unknown')
    report.append(f"Severity Class: {base_severity} → {mod_severity}")
    
    # Recommendations
    report.append("\n### RECOMMENDATIONS ###")
    for recommendation in scenario.get('recommendations', []):
        report.append(f"- {recommendation}")
    
    return "\n".join(report) 