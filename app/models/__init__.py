"""
Model modules for the SkyGuard Analytics application.
"""
# Import database models
from app.models.db.scenario import Scenario
from app.models.db.prediction_history import PredictionHistory

__all__ = ["Scenario", "PredictionHistory"]