"""
Nowcasting Schemas - Pydantic models for weather nowcasting API requests and responses.
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np


class RadarSiteInfo(BaseModel):
    """Information about a radar site."""
    site_id: str = Field(..., description="Radar site identifier (e.g., KAMX)")
    name: str = Field(..., description="Human-readable site name")
    location: str = Field(..., description="Location description")
    coordinates: List[float] = Field(..., description="[latitude, longitude]")
    description: str = Field(..., description="Site characteristics")
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError('Coordinates must be [latitude, longitude]')
        if not (-90 <= v[0] <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= v[1] <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        return v


class NowcastingPredictionRequest(BaseModel):
    """Request schema for weather nowcasting predictions."""
    site_id: str = Field(..., description="Radar site identifier (KAMX or KATX)")
    use_latest_data: bool = Field(
        default=True, 
        description="Use latest available radar data"
    )
    hours_back: int = Field(
        default=12, 
        ge=1, 
        le=48,
        description="Hours back to fetch radar data (1-48)"
    )
    custom_radar_data: Optional[List[List[List[List[float]]]]] = Field(
        default=None,
        description="Custom radar sequence data (10, 64, 64, 1) - optional"
    )
    
    @validator('site_id')
    def validate_site_id(cls, v):
        allowed_sites = ['KAMX', 'KATX']
        if v not in allowed_sites:
            raise ValueError(f'Site must be one of {allowed_sites}')
        return v
    
    @validator('custom_radar_data')
    def validate_custom_data(cls, v):
        if v is not None:
            # Convert to numpy for validation
            arr = np.array(v)
            if arr.shape != (10, 64, 64, 1):
                raise ValueError('Custom radar data must have shape (10, 64, 64, 1)')
            if arr.min() < 0 or arr.max() > 1:
                raise ValueError('Custom radar data values must be in range [0, 1]')
        return v


class NowcastingPredictionResponse(BaseModel):
    """Response schema for weather nowcasting predictions."""
    success: bool = Field(..., description="Whether prediction was successful")
    site_info: RadarSiteInfo = Field(..., description="Information about the radar site")
    prediction_frames: List[List[List[List[float]]]] = Field(
        ..., 
        description="Predicted radar frames (6, 64, 64, 1)"
    )
    input_metadata: Dict[str, Any] = Field(..., description="Information about input data")
    ml_model_metadata: Dict[str, Any] = Field(..., description="Model information")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    prediction_timestamp: datetime = Field(..., description="When prediction was generated")
    frame_times: List[str] = Field(..., description="Predicted time labels for each frame")
    confidence_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence metrics if available"
    )
    
    model_config = {
        "protected_namespaces": (),
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist()
        }
    }


class BatchNowcastingRequest(BaseModel):
    """Request schema for batch nowcasting predictions."""
    site_ids: List[str] = Field(..., description="List of radar site identifiers")
    hours_back: int = Field(
        default=12, 
        ge=1, 
        le=48,
        description="Hours back to fetch radar data (1-48)"
    )
    
    @validator('site_ids')
    def validate_site_ids(cls, v):
        allowed_sites = ['KAMX', 'KATX']
        for site_id in v:
            if site_id not in allowed_sites:
                raise ValueError(f'All sites must be one of {allowed_sites}')
        if len(v) != len(set(v)):
            raise ValueError('Duplicate site IDs not allowed')
        return v


class BatchNowcastingResponse(BaseModel):
    """Response schema for batch nowcasting predictions."""
    success: bool = Field(..., description="Whether batch processing was successful")
    predictions: Dict[str, Union[NowcastingPredictionResponse, Dict[str, str]]] = Field(
        ..., 
        description="Predictions by site_id or error information"
    )
    total_sites: int = Field(..., description="Total number of sites requested")
    successful_sites: int = Field(..., description="Number of successful predictions")
    failed_sites: int = Field(..., description="Number of failed predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    batch_timestamp: datetime = Field(..., description="When batch was processed")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class CurrentRadarConditionsResponse(BaseModel):
    """Response schema for current radar conditions."""
    site_info: RadarSiteInfo = Field(..., description="Radar site information")
    latest_data_time: Optional[datetime] = Field(
        default=None, 
        description="Timestamp of latest available data"
    )
    data_freshness_hours: Optional[float] = Field(
        default=None,
        description="How many hours old the latest data is"
    )
    available_frames: int = Field(..., description="Number of available recent frames")
    data_quality: str = Field(..., description="Data quality assessment")
    coverage_area_km: int = Field(default=150, description="Radar coverage radius in km")
    last_updated: datetime = Field(..., description="When this status was last updated")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class DataPipelineStatus(BaseModel):
    """Response schema for data pipeline status."""
    service_name: str = Field(..., description="Name of the service")
    status: str = Field(..., description="Overall service status")
    last_update: datetime = Field(..., description="Last update timestamp")
    supported_sites: List[str] = Field(..., description="List of supported radar sites")
    site_status: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Status information for each site"
    )
    storage_info: Dict[str, Any] = Field(..., description="Data storage information")
    health_checks: Dict[str, Any] = Field(..., description="Health check results")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class ModelHealthResponse(BaseModel):
    """Response schema for model health check."""
    ml_model_name: str = Field(..., description="Name of the model")
    is_loaded: bool = Field(..., description="Whether model is loaded in memory")
    ml_model_status: str = Field(..., description="Overall model status")
    last_prediction: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last successful prediction"
    )
    ml_model_info: Dict[str, Any] = Field(..., description="Model configuration and metadata")
    performance_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Recent performance metrics"
    )
    health_check_time: datetime = Field(..., description="When health check was performed")
    
    model_config = {
        "protected_namespaces": (),
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class DataRefreshRequest(BaseModel):
    """Request schema for manual data refresh."""
    site_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific sites to refresh (default: all sites)"
    )
    hours_back: int = Field(
        default=6,
        ge=1,
        le=24,
        description="How many hours back to refresh (1-24)"
    )
    force_refresh: bool = Field(
        default=False,
        description="Force refresh even if recent data exists"
    )
    
    @validator('site_ids')
    def validate_site_ids(cls, v):
        if v is not None:
            allowed_sites = ['KAMX', 'KATX']
            for site_id in v:
                if site_id not in allowed_sites:
                    raise ValueError(f'All sites must be one of {allowed_sites}')
        return v


class DataRefreshResponse(BaseModel):
    """Response schema for data refresh operation."""
    success: bool = Field(..., description="Whether refresh was successful")
    sites_refreshed: List[str] = Field(..., description="Sites that were refreshed")
    refresh_results: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Detailed results for each site"
    )
    total_files_downloaded: int = Field(..., description="Total number of files downloaded")
    total_processing_time_s: float = Field(..., description="Total processing time in seconds")
    refresh_timestamp: datetime = Field(..., description="When refresh was completed")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    success: bool = Field(default=False, description="Always false for errors")
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


# Common response models for API documentation
class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = Field(default=True, description="Operation was successful")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Optional response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }