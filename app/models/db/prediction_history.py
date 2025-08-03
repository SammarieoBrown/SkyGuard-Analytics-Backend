"""
Prediction history model for storing all predictions made by the system.
"""
from sqlalchemy import Column, String, JSON, DateTime, Float, Integer, Text
from sqlalchemy.sql import func
from app.database import Base


class PredictionHistory(Base):
    """
    Model for storing history of all predictions made by the system.
    """
    __tablename__ = "prediction_history"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Request info
    prediction_type = Column(String(50), nullable=False)  # 'property_damage', 'casualty_risk', 'severity'
    request_params = Column(JSON, nullable=False)
    
    # Event details
    event_type = Column(String(100))
    state = Column(String(10))
    magnitude = Column(Float)
    duration_hours = Column(Float)
    month = Column(Integer)
    hour = Column(Integer)
    
    # Prediction results
    prediction_result = Column(JSON, nullable=False)
    
    # Specific prediction values for easy querying
    predicted_damage = Column(Float)  # For property damage
    casualty_risk_score = Column(Float)  # For casualty risk
    risk_level = Column(Float)  # For casualty risk
    risk_category = Column(String(50))  # For casualty risk
    severity_class = Column(String(50))  # For severity
    confidence_score = Column(Float)
    
    # Performance metrics
    prediction_time_ms = Column(Integer)  # Time taken for prediction in milliseconds
    model_version = Column(String(50))
    
    # Associated scenario (if part of a scenario)
    scenario_id = Column(String(50), index=True)
    
    # Additional metadata
    extra_metadata = Column(JSON)  # Additional flexible storage
    
    def to_dict(self):
        """Convert prediction history to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "prediction_type": self.prediction_type,
            "event_type": self.event_type,
            "state": self.state,
            "magnitude": self.magnitude,
            "prediction_result": self.prediction_result,
            "confidence_score": self.confidence_score,
            "scenario_id": self.scenario_id,
        }