"""
Weather Nowcasting API Endpoints - Production endpoints for weather prediction.
"""
import os
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from app.schemas.nowcasting import (
    NowcastingPredictionRequest,
    NowcastingPredictionResponse,
    BatchNowcastingRequest,
    BatchNowcastingResponse,
    CurrentRadarConditionsResponse,
    DataPipelineStatus,
    ModelHealthResponse,
    DataRefreshRequest,
    DataRefreshResponse,
    ErrorResponse,
    SuccessResponse,
    RadarSiteInfo
)
from app.core.models.model_manager import model_manager
from app.services.nexrad_data_service import NEXRADDataService
from app.services.radar_processing_service import RadarProcessingService

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (these would typically be dependency injected)
nexrad_service = NEXRADDataService()
radar_processor = RadarProcessingService()


def get_nexrad_service():
    """Dependency to get NEXRAD data service."""
    return nexrad_service


def get_radar_processor():
    """Dependency to get radar processing service."""
    return radar_processor


@router.post("/predict", response_model=NowcastingPredictionResponse)
async def predict_weather_nowcast(
    request: NowcastingPredictionRequest,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Generate weather nowcast predictions for a specific radar site.
    
    Returns 6-frame precipitation predictions based on latest radar data or custom input.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing nowcast prediction for site {request.site_id}")
        
        # Get weather nowcasting model
        model = model_manager.get_weather_nowcasting_model()
        
        # Get site information
        site_info = RadarSiteInfo(
            site_id=request.site_id,
            **nexrad_svc.RADAR_SITES[request.site_id]
        )
        
        # Prepare input data
        if request.custom_radar_data is not None:
            # Use custom provided data
            model_input = np.array(request.custom_radar_data, dtype=np.float32)
            model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension
            
            input_metadata = {
                "data_source": "custom",
                "input_shape": list(model_input.shape),
                "data_type": "user_provided"
            }
        else:
            # Fetch and process latest radar data
            if request.use_latest_data:
                # Get recent files for the site
                recent_files = nexrad_svc.get_available_files(
                    request.site_id, 
                    hours_back=request.hours_back
                )
                
                if not recent_files:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No recent radar data available for site {request.site_id}"
                    )
                
                # Process files to create model input
                model_input, processing_metadata = processor.create_model_input_sequence(
                    recent_files[:20],  # Use up to 20 most recent files
                    sequence_length=10
                )
                
                input_metadata = {
                    "data_source": "nexrad_gcp",
                    "files_used": len(recent_files[:20]),
                    "processing_metadata": processing_metadata
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either use_latest_data must be True or custom_radar_data must be provided"
                )
        
        # Validate input data
        validation_result = processor.validate_processed_data(model_input)
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input data: {validation_result['issues']}"
            )
        
        # Generate predictions
        predictions, model_metadata = model.predict(model_input)
        
        # Convert predictions to list format for JSON response
        prediction_frames = predictions[0].tolist()  # Remove batch dimension
        
        # Calculate frame times (assuming 10-minute intervals)
        base_time = datetime.now()
        frame_times = [
            (base_time + timedelta(minutes=10 * (i + 1))).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(len(prediction_frames))
        ]
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Nowcast prediction completed for {request.site_id} in {processing_time_ms:.2f}ms")
        
        return NowcastingPredictionResponse(
            success=True,
            site_info=site_info,
            prediction_frames=prediction_frames,
            input_metadata=input_metadata,
            ml_model_metadata=model_metadata,
            processing_time_ms=processing_time_ms,
            prediction_timestamp=datetime.now(),
            frame_times=frame_times
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Nowcast prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchNowcastingResponse)
async def batch_weather_nowcast(
    request: BatchNowcastingRequest,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service)
):
    """
    Generate weather nowcast predictions for multiple radar sites.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing batch nowcast for sites: {request.site_ids}")
        
        predictions = {}
        successful_sites = 0
        failed_sites = 0
        
        for site_id in request.site_ids:
            try:
                # Create individual request
                individual_request = NowcastingPredictionRequest(
                    site_id=site_id,
                    use_latest_data=True,
                    hours_back=request.hours_back
                )
                
                # Generate prediction
                prediction = await predict_weather_nowcast(individual_request, nexrad_svc, radar_processor)
                predictions[site_id] = prediction
                successful_sites += 1
                
            except Exception as e:
                logger.error(f"Failed to process site {site_id}: {str(e)}")
                predictions[site_id] = {
                    "error": str(e),
                    "status": "failed"
                }
                failed_sites += 1
        
        total_processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchNowcastingResponse(
            success=successful_sites > 0,
            predictions=predictions,
            total_sites=len(request.site_ids),
            successful_sites=successful_sites,
            failed_sites=failed_sites,
            total_processing_time_ms=total_processing_time_ms,
            batch_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Batch nowcast failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/current-conditions/{site_id}", response_model=CurrentRadarConditionsResponse)
async def get_current_radar_conditions(
    site_id: str,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service)
):
    """
    Get current radar conditions and data availability for a site.
    """
    try:
        if site_id not in nexrad_svc.RADAR_SITES:
            raise HTTPException(
                status_code=404,
                detail=f"Site {site_id} not supported. Available sites: {list(nexrad_svc.RADAR_SITES.keys())}"
            )
        
        # Get site information
        site_info = RadarSiteInfo(
            site_id=site_id,
            **nexrad_svc.RADAR_SITES[site_id]
        )
        
        # Get available files
        recent_files = nexrad_svc.get_available_files(site_id, hours_back=24)
        
        latest_data_time = None
        data_freshness_hours = None
        data_quality = "unknown"
        
        if recent_files:
            # Get timestamp of most recent file
            latest_file = recent_files[0]  # Files are sorted by modification time
            latest_timestamp = datetime.fromtimestamp(os.path.getmtime(latest_file))
            latest_data_time = latest_timestamp
            
            # Calculate freshness
            time_diff = datetime.now() - latest_timestamp
            data_freshness_hours = time_diff.total_seconds() / 3600
            
            # Assess data quality
            if data_freshness_hours < 1:
                data_quality = "excellent"
            elif data_freshness_hours < 6:
                data_quality = "good"
            elif data_freshness_hours < 24:
                data_quality = "fair"
            else:
                data_quality = "poor"
        else:
            data_quality = "no_data"
        
        return CurrentRadarConditionsResponse(
            site_info=site_info,
            latest_data_time=latest_data_time,
            data_freshness_hours=data_freshness_hours,
            available_frames=len(recent_files),
            data_quality=data_quality,
            coverage_area_km=150,
            last_updated=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conditions for {site_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current conditions: {str(e)}"
        )


@router.get("/sites", response_model=List[RadarSiteInfo])
async def get_supported_sites(
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service)
):
    """
    Get list of supported radar sites for weather nowcasting.
    """
    try:
        sites = []
        for site_id, site_data in nexrad_svc.RADAR_SITES.items():
            sites.append(RadarSiteInfo(
                site_id=site_id,
                **site_data
            ))
        
        return sites
        
    except Exception as e:
        logger.error(f"Failed to get supported sites: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported sites: {str(e)}"
        )


@router.get("/health", response_model=ModelHealthResponse)
async def get_nowcasting_health():
    """
    Get health status of the weather nowcasting system.
    """
    try:
        # Get model health
        model = model_manager.get_weather_nowcasting_model()
        model_health = model.health_check()
        model_info = model.get_model_info()
        
        return ModelHealthResponse(
            ml_model_name="weather_nowcasting",
            is_loaded=model_health["model_loaded"],
            ml_model_status=model_health["status"],
            ml_model_info=model_info,
            health_check_time=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/data-status", response_model=DataPipelineStatus)
async def get_data_pipeline_status(
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Get status of the data pipeline and processing services.
    """
    try:
        # Get service status
        nexrad_status = nexrad_svc.get_service_status()
        processor_stats = processor.get_processing_stats()
        model_health = model_manager.get_model_health_status()
        
        return DataPipelineStatus(
            service_name="WeatherNowcastingPipeline",
            status="operational",
            last_update=datetime.now(),
            supported_sites=nexrad_status["supported_sites"],
            site_status=nexrad_status["storage_info"],
            storage_info={
                "data_directory": nexrad_status["data_dir"],
                "processor_config": processor_stats
            },
            health_checks=model_health
        )
        
    except Exception as e:
        logger.error(f"Data status check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data status check failed: {str(e)}"
        )


@router.post("/refresh-data", response_model=DataRefreshResponse)
async def refresh_radar_data(
    request: DataRefreshRequest,
    background_tasks: BackgroundTasks,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service)
):
    """
    Manually trigger refresh of radar data for specified sites.
    """
    try:
        start_time = time.time()
        
        # Determine which sites to refresh
        sites_to_refresh = request.site_ids or list(nexrad_svc.RADAR_SITES.keys())
        
        logger.info(f"Starting data refresh for sites: {sites_to_refresh}")
        
        refresh_results = {}
        total_files = 0
        
        for site_id in sites_to_refresh:
            try:
                result = nexrad_svc.download_recent_data(site_id, request.hours_back)
                refresh_results[site_id] = result
                total_files += result.get("total_files", 0)
                
            except Exception as e:
                logger.error(f"Failed to refresh data for {site_id}: {str(e)}")
                refresh_results[site_id] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        processing_time = time.time() - start_time
        
        # Schedule cleanup in background
        if not request.force_refresh:
            background_tasks.add_task(nexrad_svc.cleanup_old_data, days_to_keep=7)
        
        return DataRefreshResponse(
            success=len(refresh_results) > 0,
            sites_refreshed=list(refresh_results.keys()),
            refresh_results=refresh_results,
            total_files_downloaded=total_files,
            total_processing_time_s=processing_time,
            refresh_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Data refresh failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data refresh failed: {str(e)}"
        )


# Note: Exception handlers are registered at the app level in main.py
# Individual endpoints handle their own errors with try/catch blocks