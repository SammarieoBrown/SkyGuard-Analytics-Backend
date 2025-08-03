import logging
import uuid
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from app.core.models.model_manager import model_manager
from app.utils import numpy_to_python

logger = logging.getLogger(__name__)


class ScenarioSimulationModel:
    """
    Model for scenario simulation.
    
    This class handles scenario simulations by modifying event parameters
    and analyzing the effects on predictions.
    """
    
    def __init__(self):
        """Initialize the simulation model with prediction models from model manager."""
        self.property_damage_model = model_manager.get_property_damage_model()
        self.casualty_risk_model = model_manager.get_casualty_risk_model()
        self.severity_model = model_manager.get_severity_model()
    
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
        logger.info(f"Simulating scenario with {len(modifications)} parameter modifications")
        
        # Create a unique scenario ID
        scenario_id = f"scenario_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        try:
            # Get base predictions
            base_predictions = self._get_predictions(base_event)
            
            # Apply modifications to create modified event
            modified_event, parameter_changes = self._apply_modifications(base_event, modifications)
            
            # Get predictions for modified event
            modified_predictions = self._get_predictions(modified_event)
            
            # Analyze impact of changes
            impact_analysis = self._analyze_impact(base_predictions, modified_predictions, parameter_changes)
            
            # Calculate uncertainty metrics if requested
            confidence_intervals = None
            uncertainty_metrics = None
            
            if include_uncertainty:
                confidence_intervals = self._calculate_confidence_intervals(modified_event)
                uncertainty_metrics = self._calculate_uncertainty_metrics(modified_event)
            
            # Create result
            result = {
                "scenario_id": scenario_id,
                "base_prediction": base_predictions,
                "modified_prediction": modified_predictions,
                "parameter_changes": parameter_changes,
                "impact_analysis": impact_analysis
            }
            
            if confidence_intervals:
                result["confidence_intervals"] = confidence_intervals
                
            if uncertainty_metrics:
                result["uncertainty_metrics"] = uncertainty_metrics
                
            # Convert numpy types to Python native types
            return numpy_to_python(result)
        
        except Exception as e:
            logger.error(f"Error in scenario simulation: {str(e)}")
            # Return partial result with error information
            return {
                "scenario_id": scenario_id,
                "error": str(e),
                "base_event": base_event,
                "modifications": modifications
            }
    
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
        logger.info(f"Running batch simulation with {len(scenario_sets)} scenarios")
        
        # Create a unique batch ID
        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        try:
            # Run each scenario
            scenarios = []
            for i, modifications in enumerate(scenario_sets):
                logger.info(f"Running scenario {i+1}/{len(scenario_sets)}")
                scenario = self.simulate_scenario(
                    base_event=base_event,
                    modifications=modifications,
                    include_uncertainty=include_uncertainty
                )
                scenarios.append(scenario)
            
            # Create summary of batch results
            summary = self._create_batch_summary(scenarios)
            
            # Return result
            return {
                "batch_id": batch_id,
                "scenarios": scenarios,
                "summary": summary
            }
        
        except Exception as e:
            logger.error(f"Error in batch simulation: {str(e)}")
            return {
                "batch_id": batch_id,
                "error": str(e),
                "base_event": base_event,
                "scenario_count": len(scenario_sets)
            }
    
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
        logger.info(f"Performing sensitivity analysis on {len(parameters)} parameters")
        
        # Create a unique analysis ID
        analysis_id = f"sensitivity_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        try:
            # Get base predictions
            base_predictions = self._get_predictions(base_event)
            
            # Initialize sensitivity results
            parameter_sensitivities = {}
            visualization_data = {
                "parameter_importance": [],
                "elasticity_matrix": []
            }
            
            # Analyze each parameter
            for param in parameters:
                # Skip parameters that don't exist or aren't numeric
                if param not in base_event or not isinstance(base_event[param], (int, float)):
                    continue
                
                # Apply variation
                param_sensitivity = self._analyze_parameter_sensitivity(
                    base_event=base_event,
                    parameter=param,
                    base_predictions=base_predictions,
                    variation_range=variation_range
                )
                
                if param_sensitivity:
                    parameter_sensitivities[param] = param_sensitivity
                    
                    # Add to visualization data
                    visualization_data["parameter_importance"].append({
                        "parameter": param,
                        "importance": param_sensitivity.get("overall_importance", 0)
                    })
                    
                    elasticity_entry = {"parameter": param}
                    elasticity_entry.update({
                        k.replace("_elasticity", ""): v 
                        for k, v in param_sensitivity.items() 
                        if k.endswith("_elasticity")
                    })
                    visualization_data["elasticity_matrix"].append(elasticity_entry)
            
            # Sort parameter importance
            visualization_data["parameter_importance"].sort(
                key=lambda x: x["importance"], 
                reverse=True
            )
            
            # Return result
            return {
                "analysis_id": analysis_id,
                "base_prediction": base_predictions,
                "parameter_sensitivities": parameter_sensitivities,
                "visualization_data": visualization_data
            }
        
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            return {
                "analysis_id": analysis_id,
                "error": str(e),
                "base_event": base_event,
                "parameters": parameters
            }
    
    def _get_predictions(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions for an event.
        
        Args:
            event (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, Any]: Predictions for the event
        """
        predictions = {}
        
        # Get property damage prediction
        try:
            property_damage_result = self.property_damage_model.predict(event)
            # Extract just the key values for simulation
            predictions["property_damage"] = {
                "predicted_damage": property_damage_result.get("predicted_damage", 0.0),
                "prediction_range": property_damage_result.get("prediction_range", {})
            }
        except Exception as e:
            logger.warning(f"Error getting property damage prediction: {str(e)}")
        
        # Get casualty risk prediction
        try:
            casualty_risk_result = self.casualty_risk_model.predict(event)
            # Extract just the key values for simulation
            predictions["casualty_risk"] = {
                "casualty_risk_score": casualty_risk_result.get("casualty_risk_score", 0.0),
                "risk_level": casualty_risk_result.get("risk_level", 0.0),
                "risk_category": casualty_risk_result.get("risk_category", "Unknown")
            }
        except Exception as e:
            logger.warning(f"Error getting casualty risk prediction: {str(e)}")
        
        # Get severity prediction
        try:
            severity_result = self.severity_model.predict(event)
            # Extract just the key values for simulation
            predictions["severity"] = {
                "severity_class": severity_result.get("severity_class", "Unknown"),
                "confidence_score": severity_result.get("confidence_score", 0.0)
            }
        except Exception as e:
            logger.warning(f"Error getting severity prediction: {str(e)}")
        
        return predictions
    
    def _apply_modifications(
        self, 
        base_event: Dict[str, Any],
        modifications: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply modifications to base event.
        
        Args:
            base_event (Dict[str, Any]): Base event parameters
            modifications (List[Dict[str, Any]]): List of parameter modifications
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Modified event and parameter changes
        """
        # Create a deep copy of the base event
        modified_event = copy.deepcopy(base_event)
        
        # Track parameter changes
        parameter_changes = {}
        
        # Apply each modification
        for mod in modifications:
            parameter = mod.get("parameter")
            mod_type = mod.get("modification_type")
            value = mod.get("value")
            
            if parameter and mod_type and value is not None:
                # Skip if parameter not in base event
                if parameter not in base_event:
                    continue
                
                # Get original value
                original_value = base_event.get(parameter)
                
                # Apply modification based on type
                if mod_type == "set":
                    modified_event[parameter] = value
                elif mod_type == "add" and isinstance(original_value, (int, float)) and isinstance(value, (int, float)):
                    modified_event[parameter] = original_value + value
                elif mod_type == "multiply" and isinstance(original_value, (int, float)) and isinstance(value, (int, float)):
                    modified_event[parameter] = original_value * value
                
                # Record change
                parameter_changes[parameter] = {
                    "original": original_value,
                    "modified": modified_event[parameter],
                    "change_type": mod_type
                }
                
                # Add change factor for multiply
                if mod_type == "multiply" and isinstance(value, (int, float)):
                    parameter_changes[parameter]["change_factor"] = value
                
                # Add change amount for add
                if mod_type == "add" and isinstance(value, (int, float)):
                    parameter_changes[parameter]["change_amount"] = value
        
        return modified_event, parameter_changes
    
    def _analyze_impact(
        self,
        base_predictions: Dict[str, Any],
        modified_predictions: Dict[str, Any],
        parameter_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze impact of parameter changes on predictions.
        
        Args:
            base_predictions (Dict[str, Any]): Base predictions
            modified_predictions (Dict[str, Any]): Modified predictions
            parameter_changes (Dict[str, Any]): Parameter changes
            
        Returns:
            Dict[str, Any]: Impact analysis
        """
        impact = {}
        
        # Analyze property damage impact
        base_damage = base_predictions.get("property_damage", {}).get("predicted_damage", 0)
        mod_damage = modified_predictions.get("property_damage", {}).get("predicted_damage", 0)
        
        if base_damage and mod_damage:
            change_amount = mod_damage - base_damage
            change_percent = (change_amount / base_damage * 100) if base_damage > 0 else 0
            
            impact["property_damage_change"] = {
                "change_amount": change_amount,
                "change_percent": change_percent,
                "significance": self._get_significance(change_percent)
            }
        
        # Analyze casualty risk impact
        base_risk = base_predictions.get("casualty_risk", {}).get("casualty_risk_score", 0)
        mod_risk = modified_predictions.get("casualty_risk", {}).get("casualty_risk_score", 0)
        base_category = base_predictions.get("casualty_risk", {}).get("risk_category", "")
        mod_category = modified_predictions.get("casualty_risk", {}).get("risk_category", "")
        
        if base_risk and mod_risk:
            change_amount = mod_risk - base_risk
            change_percent = (change_amount / base_risk * 100) if base_risk > 0 else 0
            category_change = base_category != mod_category
            
            impact["casualty_risk_change"] = {
                "change_amount": change_amount,
                "change_percent": change_percent,
                "category_change": category_change,
                "significance": self._get_significance(change_percent)
            }
        
        # Analyze severity impact
        base_severity = base_predictions.get("severity", {}).get("severity_class", "")
        mod_severity = modified_predictions.get("severity", {}).get("severity_class", "")
        
        if base_severity and mod_severity:
            category_change = base_severity != mod_severity
            
            impact["severity_change"] = {
                "category_change": category_change,
                "significance": "High" if category_change else "Low"
            }
        
        # Determine overall impact
        significances = [v.get("significance") for v in impact.values() if isinstance(v, dict) and "significance" in v]
        if significances:
            if "High" in significances:
                impact["overall_impact"] = "High"
            elif "Moderate" in significances:
                impact["overall_impact"] = "Moderate"
            else:
                impact["overall_impact"] = "Low"
        
        return impact
    
    def _get_significance(self, percent_change: float) -> str:
        """
        Get significance of a percent change.
        
        Args:
            percent_change (float): Percent change
            
        Returns:
            str: Significance level
        """
        if abs(percent_change) > 50:
            return "High"
        elif abs(percent_change) > 20:
            return "Moderate"
        else:
            return "Low"
    
    def _calculate_confidence_intervals(self, event: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            event (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, Dict[str, float]]: Confidence intervals
        """
        confidence_intervals = {}
        
        # Property damage confidence intervals
        try:
            property_damage = self.property_damage_model.predict(event)
            predicted_damage = property_damage.get("predicted_damage", 0)
            
            # Use prediction range if available, otherwise use simple percentage
            prediction_range = property_damage.get("prediction_range", {})
            if isinstance(prediction_range, dict):
                lower = float(prediction_range.get("low_estimate", predicted_damage * 0.8))
                upper = float(prediction_range.get("high_estimate", predicted_damage * 1.2))
            else:
                lower = float(predicted_damage * 0.8)
                upper = float(predicted_damage * 1.2)
            
            confidence_intervals["property_damage"] = {
                "lower": lower,
                "upper": upper
            }
        except Exception:
            pass
        
        # Casualty risk confidence intervals
        try:
            casualty_risk = self.casualty_risk_model.predict(event)
            risk_score = casualty_risk.get("casualty_risk_score", 0)
            
            # Simple percentage-based intervals for risk
            lower = max(0, risk_score - risk_score * 0.15)
            upper = min(1, risk_score + risk_score * 0.15)
            
            confidence_intervals["casualty_risk"] = {
                "lower": lower,
                "upper": upper
            }
        except Exception:
            pass
        
        return confidence_intervals
    
    def _calculate_uncertainty_metrics(self, event: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate uncertainty metrics for predictions.
        
        Args:
            event (Dict[str, Any]): Event parameters
            
        Returns:
            Dict[str, float]: Uncertainty metrics
        """
        uncertainty_metrics = {}
        
        # For simplicity, use coefficient of variation as uncertainty metric
        
        # Property damage uncertainty
        try:
            property_damage = self.property_damage_model.predict(event)
            predicted_damage = property_damage.get("predicted_damage", 0)
            
            # Use prediction range to estimate standard deviation
            prediction_range = property_damage.get("prediction_range", {})
            if isinstance(prediction_range, dict):
                low = float(prediction_range.get("low_estimate", predicted_damage * 0.8))
                high = float(prediction_range.get("high_estimate", predicted_damage * 1.2))
                std_dev = (high - low) / 3.92  # Assuming 95% confidence interval
                cv = std_dev / predicted_damage if predicted_damage > 0 else 0
                uncertainty_metrics["property_damage_cv"] = float(cv)
            else:
                # Default uncertainty if prediction_range is not available
                uncertainty_metrics["property_damage_cv"] = 0.2
        except Exception:
            pass
        
        # Casualty risk uncertainty
        try:
            casualty_risk = self.casualty_risk_model.predict(event)
            risk_score = casualty_risk.get("casualty_risk_score", 0)
            
            # For probabilistic outputs, uncertainty depends on proximity to 0.5
            uncertainty = 1 - 2 * abs(risk_score - 0.5)
            uncertainty_metrics["casualty_risk_uncertainty"] = uncertainty
        except Exception:
            pass
        
        return uncertainty_metrics
    
    def _create_batch_summary(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of batch simulation results.
        
        Args:
            scenarios (List[Dict[str, Any]]): List of scenario results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        summary = {
            "scenario_count": len(scenarios),
            "parameters_modified": set(),
            "significant_scenarios": 0,
            "property_damage_range": {
                "min": float('inf'),
                "max": float('-inf'),
                "mean": 0
            },
            "casualty_risk_range": {
                "min": float('inf'),
                "max": float('-inf'),
                "mean": 0
            }
        }
        
        # Collect statistics
        damage_values = []
        risk_values = []
        
        for scenario in scenarios:
            # Count parameters modified
            parameter_changes = scenario.get("parameter_changes", {})
            summary["parameters_modified"].update(parameter_changes.keys())
            
            # Count significant scenarios
            impact_analysis = scenario.get("impact_analysis", {})
            if impact_analysis.get("overall_impact") == "High":
                summary["significant_scenarios"] += 1
            
            # Collect damage values
            mod_predictions = scenario.get("modified_prediction", {})
            property_damage = mod_predictions.get("property_damage", {})
            if property_damage:
                damage = property_damage.get("predicted_damage")
                if damage is not None:
                    damage_values.append(damage)
                    summary["property_damage_range"]["min"] = min(summary["property_damage_range"]["min"], damage)
                    summary["property_damage_range"]["max"] = max(summary["property_damage_range"]["max"], damage)
            
            # Collect risk values
            casualty_risk = mod_predictions.get("casualty_risk", {})
            if casualty_risk:
                risk = casualty_risk.get("casualty_risk_score")
                if risk is not None:
                    risk_values.append(risk)
                    summary["casualty_risk_range"]["min"] = min(summary["casualty_risk_range"]["min"], risk)
                    summary["casualty_risk_range"]["max"] = max(summary["casualty_risk_range"]["max"], risk)
        
        # Calculate means
        if damage_values:
            summary["property_damage_range"]["mean"] = sum(damage_values) / len(damage_values)
        else:
            summary["property_damage_range"] = None
            
        if risk_values:
            summary["casualty_risk_range"]["mean"] = sum(risk_values) / len(risk_values)
        else:
            summary["casualty_risk_range"] = None
        
        # Convert set to list
        summary["parameters_modified"] = list(summary["parameters_modified"])
        
        return summary
    
    def _analyze_parameter_sensitivity(
        self,
        base_event: Dict[str, Any],
        parameter: str,
        base_predictions: Dict[str, Any],
        variation_range: float
    ) -> Optional[Dict[str, float]]:
        """
        Analyze sensitivity of a single parameter.
        
        Args:
            base_event (Dict[str, Any]): Base event parameters
            parameter (str): Parameter to analyze
            base_predictions (Dict[str, Any]): Base predictions
            variation_range (float): Variation range
            
        Returns:
            Optional[Dict[str, float]]: Sensitivity metrics for the parameter
        """
        # Skip if parameter is not numeric
        if not isinstance(base_event.get(parameter), (int, float)):
            return None
        
        # Extract base values
        base_damage = base_predictions.get("property_damage", {}).get("predicted_damage", 0)
        base_risk = base_predictions.get("casualty_risk", {}).get("casualty_risk_score", 0)
        base_value = base_event.get(parameter, 0)
        
        # Skip if base value is 0
        if base_value == 0:
            return None
        
        # Create upper and lower variations
        upper_value = base_value * (1 + variation_range)
        lower_value = base_value * (1 - variation_range)
        
        # Get predictions for upper variation
        upper_event = copy.deepcopy(base_event)
        upper_event[parameter] = upper_value
        upper_predictions = self._get_predictions(upper_event)
        
        # Get predictions for lower variation
        lower_event = copy.deepcopy(base_event)
        lower_event[parameter] = lower_value
        lower_predictions = self._get_predictions(lower_event)
        
        # Calculate elasticities
        sensitivity = {}
        
        # Property damage elasticity
        upper_damage = upper_predictions.get("property_damage", {}).get("predicted_damage", 0)
        lower_damage = lower_predictions.get("property_damage", {}).get("predicted_damage", 0)
        
        if base_damage and upper_damage and lower_damage:
            upper_pct_change = (upper_damage - base_damage) / base_damage
            lower_pct_change = (base_damage - lower_damage) / base_damage
            pct_input_change = variation_range
            
            # Average elasticity
            if pct_input_change > 0:
                elasticity = (upper_pct_change + lower_pct_change) / (2 * pct_input_change)
                sensitivity["property_damage_elasticity"] = elasticity
        
        # Casualty risk elasticity
        upper_risk = upper_predictions.get("casualty_risk", {}).get("casualty_risk_score", 0)
        lower_risk = lower_predictions.get("casualty_risk", {}).get("casualty_risk_score", 0)
        
        if base_risk and upper_risk and lower_risk:
            upper_pct_change = (upper_risk - base_risk) / base_risk
            lower_pct_change = (base_risk - lower_risk) / base_risk
            pct_input_change = variation_range
            
            # Average elasticity
            if pct_input_change > 0:
                elasticity = (upper_pct_change + lower_pct_change) / (2 * pct_input_change)
                sensitivity["casualty_risk_elasticity"] = elasticity
        
        # Calculate overall importance
        elasticities = [
            abs(v) for k, v in sensitivity.items() if k.endswith("_elasticity")
        ]
        
        if elasticities:
            sensitivity["overall_importance"] = sum(elasticities) / len(elasticities)
        
        return sensitivity 