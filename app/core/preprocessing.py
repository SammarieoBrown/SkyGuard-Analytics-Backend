"""
Preprocessing classes for SkyGuard Analytics models.

This module contains preprocessing classes that are required for loading 
serialized model files. The classes here mirror the essential structure
of those used during model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PropertyDamageDataPreprocessor:
    """Handles data preprocessing for property damage prediction."""
    
    def __init__(self):
        self.label_encoders = {}
        self.column_transformer = None
        self._name_to_fitted_passthrough = {}  # Fix for sklearn compatibility
        self.scaler = None
        self.feature_names = None
        
    def transform(self, X):
        """Transform input data."""
        return X
        
    def fit_transform(self, X, y=None):
        """Fit and transform input data."""
        return X
        
    def fit(self, X, y=None):
        """Fit the preprocessor."""
        return self


class CasualtyRiskPreprocessor:
    """Handles data preprocessing for casualty risk classification."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        
    def create_target_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable for casualty occurrence.
        
        Args:
            df: DataFrame with injury and death columns
            
        Returns:
            Binary series indicating casualty occurrence
        """
        # Handle missing values
        injuries = df['injuries_direct'].fillna(0) + df['injuries_indirect'].fillna(0)
        deaths = df['deaths_direct'].fillna(0) + df['deaths_indirect'].fillna(0)
        
        # Binary target: 1 if any casualties, 0 otherwise
        return (injuries + deaths > 0).astype(int)
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepare features for casualty risk modeling.
        
        Args:
            df: Raw DataFrame
            is_training: Whether this is training data
            
        Returns:
            Prepared feature DataFrame
        """
        features_df = df.copy()
        
        # Handle missing values in key features
        features_df['magnitude'] = features_df['magnitude'].fillna(-1)
        features_df['event_duration_hours'] = (
            (pd.to_datetime(df['event_end_time'], errors='coerce') - 
             pd.to_datetime(df['event_begin_time'], errors='coerce')).dt.total_seconds() / 3600
        ).fillna(0)
        
        # Cap extreme durations
        features_df.loc[features_df['event_duration_hours'] > 168, 'event_duration_hours'] = 168  # 1 week max
        features_df.loc[features_df['event_duration_hours'] < 0, 'event_duration_hours'] = 0
        
        # Extract temporal features
        features_df['event_datetime'] = pd.to_datetime(df['event_begin_time'], errors='coerce')
        features_df['hour'] = features_df['event_datetime'].dt.hour
        features_df['day_of_week'] = features_df['event_datetime'].dt.dayofweek
        features_df['month'] = features_df['event_datetime'].dt.month
        features_df['year'] = features_df['event_datetime'].dt.year
        
        # Time-based risk factors
        features_df['is_night'] = features_df['hour'].between(20, 6).astype(int)
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_rush_hour'] = (
            features_df['hour'].between(7, 9) | features_df['hour'].between(17, 19)
        ).astype(int)
        
        # Season
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        features_df['season'] = features_df['month'].map(season_map)
        
        # Event type encoding
        if is_training:
            le = LabelEncoder()
            features_df['event_type_encoded'] = le.fit_transform(features_df['event_type'].fillna('Unknown'))
            self.label_encoders['event_type'] = le
        else:
            le = self.label_encoders['event_type']
            # Handle unseen categories
            features_df['event_type_encoded'] = features_df['event_type'].fillna('Unknown').map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        
        # State encoding
        if is_training:
            le_state = LabelEncoder()
            features_df['state_encoded'] = le_state.fit_transform(features_df['state'].fillna('Unknown'))
            self.label_encoders['state'] = le_state
        else:
            le_state = self.label_encoders['state']
            features_df['state_encoded'] = features_df['state'].fillna('Unknown').map(
                lambda x: le_state.transform([x])[0] if x in le_state.classes_ else -1
            )
        
        return features_df


class CasualtyFeatureEngineer:
    """Create specialized features for casualty risk prediction."""
    
    def __init__(self):
        self.event_casualty_stats = {}
        self.state_casualty_stats = {}
        self.population_data = None
        
    def create_severity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create event severity indicators.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with severity features
        """
        df = df.copy()
        
        # Magnitude-based severity (handle missing values)
        df['has_magnitude'] = (~df['magnitude'].isna() & (df['magnitude'] > 0)).astype(int)
        df['magnitude_severity'] = pd.cut(
            df['magnitude'].fillna(0),
            bins=[-np.inf, 0, 50, 75, 100, np.inf],
            labels=['none', 'low', 'moderate', 'high', 'extreme']
        )
        
        # Duration-based severity
        df['duration_severity'] = pd.cut(
            df['event_duration_hours'],
            bins=[0, 1, 6, 24, 168, np.inf],
            labels=['brief', 'short', 'moderate', 'long', 'extended']
        )
        
        # Tornado-specific features
        df['is_tornado'] = df['event_type'].str.lower().str.contains('tornado', na=False).astype(int)
        
        # Map F/EF scale to numeric
        f_scale_map = {
            'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3, 'F4': 4, 'F5': 5,
            'EF0': 0, 'EF1': 1, 'EF2': 2, 'EF3': 3, 'EF4': 4, 'EF5': 5
        }
        df['tornado_scale'] = df['tor_f_scale'].map(f_scale_map).fillna(0)
        
        # High-risk event types
        high_risk_events = ['tornado', 'flash flood', 'hurricane', 'wildfire', 'avalanche']
        df['is_high_risk_event'] = df['event_type'].str.lower().apply(
            lambda x: any(event in str(x) for event in high_risk_events) if pd.notna(x) else False
        ).astype(int)
        
        return df
    
    def create_population_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to population at risk.
        
        Args:
            df: DataFrame with location data
            
        Returns:
            DataFrame with population risk features
        """
        df = df.copy()
        
        # State-level population density indicators (simplified)
        # In production, would load actual population data
        high_density_states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
        df['is_high_density_state'] = df['state'].isin(high_density_states).astype(int)
        
        # Urban vs rural approximation based on location description
        urban_indicators = ['city', 'metro', 'downtown', 'urban']
        df['is_urban'] = df['cz_name'].str.lower().apply(
            lambda x: any(ind in str(x) for ind in urban_indicators) if pd.notna(x) else False
        ).astype(int)
        
        # Time of day population exposure
        df['daytime_population_risk'] = (
            (df['hour'].between(9, 17)) & (df['day_of_week'].between(0, 4))
        ).astype(int)
        
        return df
    
    def create_historical_patterns(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create features based on historical casualty patterns.
        
        Args:
            df: DataFrame with event data
            is_training: Whether to fit statistics
            
        Returns:
            DataFrame with historical pattern features
        """
        df = df.copy()
        
        if is_training:
            # Calculate casualty statistics by event type
            casualty_indicator = ((df['injuries_direct'].fillna(0) + df['injuries_indirect'].fillna(0) +
                                 df['deaths_direct'].fillna(0) + df['deaths_indirect'].fillna(0)) > 0)
            
            self.event_casualty_stats = df.groupby('event_type').agg({
                'injuries_direct': ['mean', 'max'],
                'deaths_direct': ['mean', 'max']
            }).fillna(0)
            
            # Casualty probability by event type
            event_casualty_prob = casualty_indicator.groupby(df['event_type']).mean()
            df['event_casualty_prob'] = df['event_type'].map(event_casualty_prob).fillna(0)
            
            # State-level statistics
            self.state_casualty_stats = df.groupby('state').agg({
                'injuries_direct': ['mean', 'sum'],
                'deaths_direct': ['mean', 'sum']
            }).fillna(0)
            
            state_casualty_prob = casualty_indicator.groupby(df['state']).mean()
            df['state_casualty_prob'] = df['state'].map(state_casualty_prob).fillna(0)
        else:
            # Use stored statistics
            df['event_casualty_prob'] = df['event_type'].map(
                lambda x: self.event_casualty_stats.get(x, 0) if pd.notna(x) else 0
            )
            df['state_casualty_prob'] = df['state'].map(
                lambda x: self.state_casualty_stats.get(x, 0) if pd.notna(x) else 0
            )
        
        # Temporal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key risk factors.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # High-risk combinations
        df['tornado_night'] = df['is_tornado'] * df['is_night']
        df['high_risk_urban'] = df['is_high_risk_event'] * df['is_urban']
        df['severe_magnitude_duration'] = (
            (df['magnitude'] > 75) & (df['event_duration_hours'] > 6)
        ).astype(int)
        
        # Population exposure interactions
        df['urban_rush_hour'] = df['is_urban'] * df['is_rush_hour']
        df['weekend_severity'] = df['is_weekend'] * df['has_magnitude']
        
        # State-event risk combination
        df['high_risk_state_event'] = df['is_high_density_state'] * df['is_high_risk_event']
        
        return df


class FocalLoss:
    """
    Focal Loss for addressing class imbalance in casualty prediction.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor in range [0, 1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
        """
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_pred: np.ndarray, y_true) -> tuple:
        """
        Calculate focal loss gradient and hessian for XGBoost.
        
        Args:
            y_pred: Predictions from previous iteration (raw logits)
            y_true: True labels from DMatrix
            
        Returns:
            Gradient and hessian of focal loss
        """
        # This is a placeholder implementation
        # The actual implementation would be used during training
        # Not needed for inference
        pass


class DummyLoss:
    """
    Dummy loss class for handling unsupported loss functions.
    """
    
    def __init__(self, *args, **kwargs):
        pass
        
    def __call__(self, *args, **kwargs):
        return 0.0
        
    def __reduce__(self):
        return (type(self), ())
        
    def __getstate__(self):
        return {}
        
    def __setstate__(self, state):
        pass


class SeverityClassifier:
    """
    Classifier for weather event severity levels.
    This is a placeholder for compatibility with pickled models.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the severity classifier."""
        self.classes_ = ['Minor', 'Moderate', 'Significant', 'Severe', 'Catastrophic']
        self.n_classes_ = len(self.classes_)
        
    def predict(self, X):
        """
        Predict severity class for input features.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted severity classes
        """
        # Return default predictions
        n_samples = len(X) if hasattr(X, '__len__') else 1
        return np.array(['Moderate'] * n_samples)
    
    def predict_proba(self, X):
        """
        Predict probability distribution over severity classes.
        
        Args:
            X: Input features
            
        Returns:
            Array of probability distributions
        """
        n_samples = len(X) if hasattr(X, '__len__') else 1
        # Return uniform probabilities
        proba = np.ones((n_samples, self.n_classes_)) / self.n_classes_
        return proba
    
    def __getstate__(self):
        """Get state for pickling."""
        return self.__dict__
    
    def __setstate__(self, state):
        """Set state for unpickling."""
        self.__dict__.update(state)
        pass 