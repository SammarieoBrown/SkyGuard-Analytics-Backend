from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime
from app.core.models.simulation_model import ScenarioSimulationModel
from app.database import SessionLocal
from app.models import Scenario as ScenarioModel
from app.utils import numpy_to_python

logger = logging.getLogger(__name__)


class ScenarioSimulationService:
    """
    Service for scenario simulations.
    This service acts as an intermediary between the API and the underlying simulation model.
    """
    
    def __init__(self):
        """Initialize service with required models."""
        self.simulation_model = ScenarioSimulationModel()
    
    def simulate_scenario(
        self, 
        base_event: Dict[str, Any],
        modifications: List[Dict[str, Any]],
        include_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate a scenario by modifying event parameters.
        
        Args:
            base_event (Dict[str, Any]): Base event parameters
            modifications (List[Dict[str, Any]]): List of parameter modifications
            include_uncertainty (bool): Include uncertainty metrics in the result
            
        Returns:
            Dict[str, Any]: Simulation result
        """
        logger.info(f"Service: Simulating scenario with {len(modifications)} parameter modifications")
        
        try:
            result = self.simulation_model.simulate_scenario(
                base_event=base_event,
                modifications=modifications,
                include_uncertainty=include_uncertainty
            )
            
            # Add recommendations based on results
            result['recommendations'] = self._generate_recommendations(base_event, result)
            
            # Save scenario to database
            self._save_scenario(result, scenario_type="single")
            
            return result
        except Exception as e:
            logger.error(f"Error in scenario simulation service: {str(e)}")
            raise
    
    def batch_simulate(
        self, 
        base_event: Dict[str, Any],
        scenario_sets: List[List[Dict[str, Any]]],
        include_uncertainty: bool = False
    ) -> Dict[str, Any]:
        """
        Run batch scenario simulations.
        
        Args:
            base_event (Dict[str, Any]): Base event parameters
            scenario_sets (List[List[Dict[str, Any]]]): List of modification sets
            include_uncertainty (bool): Include uncertainty metrics in the results
            
        Returns:
            Dict[str, Any]: Batch simulation results
        """
        logger.info(f"Service: Running batch simulation with {len(scenario_sets)} scenarios")
        
        try:
            result = self.simulation_model.batch_simulate(
                base_event=base_event,
                scenario_sets=scenario_sets,
                include_uncertainty=include_uncertainty
            )
            
            # Add recommendations for each scenario
            for scenario in result.get('scenarios', []):
                scenario['recommendations'] = self._generate_recommendations(base_event, scenario)
            
            # Add batch-level recommendations
            result['recommendations'] = self._generate_batch_recommendations(result)
            
            # Save scenario to database
            self._save_scenario(result, scenario_type="batch")
            
            return result
        except Exception as e:
            logger.error(f"Error in batch simulation service: {str(e)}")
            raise
    
    def perform_sensitivity_analysis(
        self, 
        base_event: Dict[str, Any],
        parameters: List[str],
        variation_range: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on selected parameters.
        
        Args:
            base_event (Dict[str, Any]): Base event parameters
            parameters (List[str]): Parameters to analyze
            variation_range (float): Variation range for parameter values
            
        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        logger.info(f"Service: Performing sensitivity analysis on {len(parameters)} parameters")
        
        try:
            result = self.simulation_model.perform_sensitivity_analysis(
                base_event=base_event,
                parameters=parameters,
                variation_range=variation_range
            )
            
            # Add recommendations based on sensitivity analysis
            result['recommendations'] = self._generate_sensitivity_recommendations(result)
            
            # Save scenario to database
            self._save_scenario(result, scenario_type="sensitivity")
            
            return result
        except Exception as e:
            logger.error(f"Error in sensitivity analysis service: {str(e)}")
            raise
    
    def get_saved_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a saved scenario by ID from the database.
        
        Args:
            scenario_id (str): Scenario ID
            
        Returns:
            Optional[Dict[str, Any]]: Saved scenario or None if not found
        """
        logger.info(f"Service: Getting saved scenario {scenario_id}")
        
        db = SessionLocal()
        try:
            scenario = db.query(ScenarioModel).filter(
                ScenarioModel.scenario_id == scenario_id
            ).first()
            
            if scenario:
                logger.info(f"Found scenario {scenario_id} in database")
                return scenario.to_dict()
            else:
                logger.warning(f"Scenario {scenario_id} not found in database")
                return None
        except Exception as e:
            logger.error(f"Error retrieving scenario {scenario_id}: {str(e)}")
            return None
        finally:
            db.close()
    
    def _generate_recommendations(self, base_event: Dict[str, Any], scenario_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on scenario results.
        
        Args:
            base_event (Dict[str, Any]): Base event parameters
            scenario_result (Dict[str, Any]): Scenario result
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Check if we have both base and modified predictions
        base_pred = scenario_result.get('base_prediction', {})
        mod_pred = scenario_result.get('modified_prediction', {})
        impact = scenario_result.get('impact_analysis', {})
        
        # Property damage recommendations
        base_damage = base_pred.get('property_damage', {}).get('predicted_damage', 0)
        mod_damage = mod_pred.get('property_damage', {}).get('predicted_damage', 0)
        
        if mod_damage > base_damage * 1.5:
            recommendations.append(f"Anticipate significant increase in property damage claims ({int((mod_damage-base_damage)/1000)}k USD increase)")
        
        # Casualty risk recommendations
        base_risk = base_pred.get('casualty_risk', {}).get('casualty_risk_score', 0)
        mod_risk = mod_pred.get('casualty_risk', {}).get('casualty_risk_score', 0)
        base_category = base_pred.get('casualty_risk', {}).get('risk_category', '')
        mod_category = mod_pred.get('casualty_risk', {}).get('risk_category', '')
        
        if mod_risk > base_risk * 1.3 or (base_category != mod_category and mod_category in ['High', 'Very High']):
            recommendations.append("Consider expanding evacuation zones due to increased casualty risk")
            
        # Severity recommendations
        base_severity = base_pred.get('severity', {}).get('severity_class', '')
        mod_severity = mod_pred.get('severity', {}).get('severity_class', '')
        
        severity_escalation = {
            'Minor': 1,
            'Moderate': 2,
            'Significant': 3,
            'Severe': 4,
            'Catastrophic': 5
        }
        
        if (severity_escalation.get(mod_severity, 0) > severity_escalation.get(base_severity, 0)):
            recommendations.append(f"Alert emergency services about potential severity escalation to {mod_severity}")
            
            if mod_severity in ['Severe', 'Catastrophic']:
                recommendations.append("Activate emergency response protocols for high-severity weather events")
        
        # Overall recommendations
        if impact.get('overall_impact') == 'High':
            event_type = base_event.get('event_type', 'weather event')
            recommendations.append(f"Update risk assessment models for {event_type} to account for parameter sensitivities")
            
        # Return at least one recommendation even if none of the above conditions are met
        if not recommendations:
            recommendations.append("Monitor situation for changes in forecast parameters")
            
        return recommendations
    
    def _generate_batch_recommendations(self, batch_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on batch simulation results.
        
        Args:
            batch_result (Dict[str, Any]): Batch simulation result
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Extract summary statistics
        summary = batch_result.get('summary', {})
        significant_scenarios = summary.get('significant_scenarios', 0)
        total_scenarios = summary.get('scenario_count', 0)
        parameters_modified = summary.get('parameters_modified', [])
        
        # Generate recommendations based on statistics
        if significant_scenarios / total_scenarios > 0.5:
            recommendations.append("High proportion of significant impact scenarios detected - increase monitoring frequency")
            
        # Parameter-specific recommendations
        if 'magnitude' in parameters_modified:
            recommendations.append("Event magnitude is a key sensitivity factor - verify forecasts for accuracy")
            
        if 'tor_f_scale' in parameters_modified:
            recommendations.append("Tornado scale has major impact - prepare for potential scale changes")
            
        # General recommendations
        recommendations.append(f"Review {len(parameters_modified)} key parameters for improved forecast accuracy")
        
        # Return at least one recommendation even if none of the above conditions are met
        if not recommendations:
            recommendations.append("Continue batch scenario analysis to identify key risk factors")
            
        return recommendations
    
    def _generate_sensitivity_recommendations(self, sensitivity_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on sensitivity analysis.
        
        Args:
            sensitivity_result (Dict[str, Any]): Sensitivity analysis result
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Extract parameter sensitivities
        parameter_sensitivities = sensitivity_result.get('parameter_sensitivities', {})
        
        # Sort parameters by overall importance
        sorted_parameters = sorted(
            parameter_sensitivities.items(),
            key=lambda x: x[1].get('overall_importance', 0),
            reverse=True
        )
        
        # Generate recommendations for top parameters
        if sorted_parameters:
            top_param, top_data = sorted_parameters[0]
            importance = top_data.get('overall_importance', 0)
            
            if importance > 0.5:
                recommendations.append(f"Focus monitoring on {top_param} as it has the highest impact on predictions")
            
            # Add recommendations based on elasticity
            damage_elasticity = top_data.get('property_damage_elasticity', 0)
            risk_elasticity = top_data.get('casualty_risk_elasticity', 0)
            
            if abs(damage_elasticity) > 1:
                recommendations.append(f"Small changes in {top_param} can cause large changes in property damage - verify measurements")
                
            if abs(risk_elasticity) > 1:
                recommendations.append(f"Casualty risk is highly sensitive to {top_param} - prioritize in updates")
        
        # Add general recommendations based on visualization data
        vis_data = sensitivity_result.get('visualization_data', {})
        param_importance = vis_data.get('parameter_importance', [])
        
        if len(param_importance) > 1:
            top_params = [p['parameter'] for p in param_importance[:2]]
            recommendations.append(f"Prioritize accurate forecasting of {' and '.join(top_params)} for most reliable predictions")
        
        # Return at least one recommendation
        if not recommendations:
            recommendations.append("Continue sensitivity analysis with more parameters to identify key factors")
            
        return recommendations
    
    def _save_scenario(self, result: Dict[str, Any], scenario_type: str) -> None:
        """
        Save scenario result to database.
        
        Args:
            result (Dict[str, Any]): Scenario result
            scenario_type (str): Type of scenario ('single', 'batch', 'sensitivity')
        """
        db = SessionLocal()
        try:
            # Convert numpy types to Python native types before saving
            scenario = ScenarioModel(
                scenario_id=result.get('scenario_id') or result.get('batch_id') or result.get('analysis_id'),
                scenario_type=scenario_type,
                base_event=numpy_to_python(result.get('base_event')),
                modifications=numpy_to_python(result.get('modifications')),
                scenario_sets=numpy_to_python(result.get('scenario_sets')),
                parameters=numpy_to_python(result.get('parameters')),
                base_prediction=numpy_to_python(result.get('base_prediction')),
                modified_prediction=numpy_to_python(result.get('modified_prediction')),
                scenarios=numpy_to_python(result.get('scenarios')),
                parameter_sensitivities=numpy_to_python(result.get('parameter_sensitivities')),
                parameter_changes=numpy_to_python(result.get('parameter_changes')),
                impact_analysis=numpy_to_python(result.get('impact_analysis')),
                confidence_intervals=numpy_to_python(result.get('confidence_intervals')),
                uncertainty_metrics=numpy_to_python(result.get('uncertainty_metrics')),
                visualization_data=numpy_to_python(result.get('visualization_data')),
                summary=numpy_to_python(result.get('summary')),
                recommendations=numpy_to_python(result.get('recommendations'))
            )
            
            db.add(scenario)
            db.commit()
            logger.info(f"Saved {scenario_type} scenario {scenario.scenario_id} to database")
        except Exception as e:
            logger.error(f"Error saving scenario to database: {str(e)}")
            db.rollback()
        finally:
            db.close() 