"""
Scenario model for storing simulation scenarios.
"""
from sqlalchemy import Column, String, JSON, DateTime, Float, Text
from sqlalchemy.sql import func
from app.database import Base


class Scenario(Base):
    """
    Model for storing weather event scenarios and their predictions.
    """
    __tablename__ = "scenarios"
    
    # Primary key
    scenario_id = Column(String(50), primary_key=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Scenario type
    scenario_type = Column(String(50), nullable=False)  # 'single', 'batch', 'sensitivity'
    
    # Event data
    base_event = Column(JSON, nullable=False)
    modifications = Column(JSON)  # For single scenarios
    scenario_sets = Column(JSON)  # For batch scenarios
    parameters = Column(JSON)  # For sensitivity analysis
    
    # Predictions
    base_prediction = Column(JSON)
    modified_prediction = Column(JSON)  # For single scenarios
    scenarios = Column(JSON)  # For batch scenarios
    parameter_sensitivities = Column(JSON)  # For sensitivity analysis
    
    # Analysis results
    parameter_changes = Column(JSON)
    impact_analysis = Column(JSON)
    confidence_intervals = Column(JSON)
    uncertainty_metrics = Column(JSON)
    visualization_data = Column(JSON)
    
    # Summary data
    summary = Column(JSON)
    recommendations = Column(JSON)
    
    # Additional metadata
    extra_metadata = Column(JSON)  # Additional flexible storage
    
    def to_dict(self):
        """Convert scenario to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "scenario_type": self.scenario_type,
            "base_event": self.base_event,
            "modifications": self.modifications,
            "base_prediction": self.base_prediction,
            "modified_prediction": self.modified_prediction,
            "parameter_changes": self.parameter_changes,
            "impact_analysis": self.impact_analysis,
            "confidence_intervals": self.confidence_intervals,
            "uncertainty_metrics": self.uncertainty_metrics,
            "recommendations": self.recommendations,
        }