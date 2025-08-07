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
from fastapi.responses import JSONResponse, Response

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
    RadarSiteInfo,
    # New schemas for raw radar data
    RawRadarDataResponse,
    TimeSeriesRadarResponse,
    MultiSiteRadarResponse,
    RadarDataFrame,
    CoordinateMetadata,
    RadarFrameRequest,
    TimeSeriesRequest
)
from app.core.models.model_manager import model_manager
from app.services.nexrad_data_service import NEXRADDataService
from app.services.radar_processing_service import RadarProcessingService
from app.services.radar_mosaic_service import RadarMosaicService
from app.utils.coordinate_utils import create_coordinate_metadata, get_site_coordinates
from app.config import RADAR_MAX_BATCH_SIZE, IS_RENDER

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (these would typically be dependency injected)
nexrad_service = NEXRADDataService()
radar_processor = RadarProcessingService()
mosaic_service = RadarMosaicService()


def get_nexrad_service():
    """Dependency to get NEXRAD data service."""
    return nexrad_service


def get_radar_processor():
    """Dependency to get radar processing service."""
    return radar_processor


def get_mosaic_service():
    """Dependency to get radar mosaic service."""
    return mosaic_service


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
            # Fetch and process latest radar data (optimized for speed)
            if request.use_latest_data:
                # Try to get existing files first, then download if needed
                recent_files = nexrad_svc.get_available_files(
                    request.site_id, 
                    hours_back=request.hours_back,
                    max_files=20  # Limit for faster processing
                )
                
                # If no files available, download latest files optimally
                if not recent_files:
                    logger.info(f"No local files for {request.site_id}, downloading latest data")
                    download_result = nexrad_svc.download_latest_files(request.site_id, max_files=20)
                    recent_files = download_result.get("file_list", [])
                
                if not recent_files:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No recent radar data available for site {request.site_id}"
                    )
                
                # Limit files for memory constraints on Render
                max_files = min(20, RADAR_MAX_BATCH_SIZE + 5)  # Need extra for sequence
                
                # Process files to create model input with enhanced performance
                model_input, processing_metadata = processor.create_model_input_sequence(
                    recent_files[:max_files],  # Limit files for memory
                    sequence_length=10,
                    site_id=request.site_id,
                    use_cache=True,
                    concurrent=not IS_RENDER  # Sequential on Render, concurrent locally
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


@router.post("/warm-cache", response_model=SuccessResponse)
async def warm_cache_for_site(
    site_id: str,
    hours_back: int = 6,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Warm cache with recent radar data for faster predictions.
    """
    try:
        if site_id not in nexrad_svc.RADAR_SITES:
            raise HTTPException(
                status_code=404,
                detail=f"Site {site_id} not supported. Available sites: {list(nexrad_svc.RADAR_SITES.keys())}"
            )
        
        # Get recent files
        recent_files = nexrad_svc.get_available_files(site_id, hours_back)
        
        if not recent_files:
            # Download recent data if not available
            logger.info(f"No local data for {site_id}, downloading recent data")
            download_result = nexrad_svc.download_recent_data(site_id, hours_back)
            recent_files = nexrad_svc.get_available_files(site_id, hours_back)
        
        # Warm cache
        warming_result = processor.warm_cache(site_id, recent_files)
        
        return SuccessResponse(
            message=f"Cache warmed for site {site_id}",
            data={
                "site_id": site_id,
                "files_processed": len(recent_files),
                "warming_result": warming_result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache warming failed for {site_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache warming failed: {str(e)}"
        )


@router.get("/cache-stats", response_model=Dict[str, Any])
async def get_cache_statistics(
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Get cache performance statistics.
    """
    try:
        stats = processor.get_processing_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@router.get("/visualization/{site_id}", response_class=Response)
async def get_radar_visualization(
    site_id: str,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor),
    mosaic_svc: RadarMosaicService = Depends(get_mosaic_service)
):
    """
    Generate NWS-style radar visualization for a specific site.
    
    Returns a PNG image of the radar data overlaid on a geographic map.
    """
    try:
        if site_id not in nexrad_svc.RADAR_SITES:
            raise HTTPException(
                status_code=404,
                detail=f"Site {site_id} not supported. Available sites: {list(nexrad_svc.RADAR_SITES.keys())}"
            )
        
        # Get recent files
        recent_files = nexrad_svc.get_available_files(site_id, hours_back=6, max_files=1)
        
        if not recent_files:
            # Download latest data if not available
            download_result = nexrad_svc.download_latest_files(site_id, max_files=1)
            recent_files = download_result.get("file_list", [])
        
        if not recent_files:
            raise HTTPException(
                status_code=404,
                detail=f"No radar data available for site {site_id}"
            )
        
        # Process the most recent file
        processed_frame = processor.process_nexrad_file(recent_files[0], site_id)
        
        if processed_frame is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process radar data for site {site_id}"
            )
        
        # Generate visualization
        image_data = mosaic_svc.create_site_visualization(site_id, processed_frame)
        
        return Response(
            content=image_data,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={site_id}_radar.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed for {site_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Visualization generation failed: {str(e)}"
        )


@router.get("/mosaic", response_class=Response)
async def get_radar_mosaic(
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor),
    mosaic_svc: RadarMosaicService = Depends(get_mosaic_service)
):
    """
    Generate composite radar mosaic from all available sites.
    
    Creates a NWS-style composite radar image showing precipitation 
    from multiple radar sites overlaid on a geographic map.
    """
    try:
        site_data = {}
        
        # Process data from all supported sites
        for site_id in nexrad_svc.RADAR_SITES.keys():
            try:
                # Get recent files for this site
                recent_files = nexrad_svc.get_available_files(site_id, hours_back=6, max_files=1)
                
                if not recent_files:
                    # Try to download latest data
                    download_result = nexrad_svc.download_latest_files(site_id, max_files=1)
                    recent_files = download_result.get("file_list", [])
                
                if recent_files:
                    # Process the most recent file
                    processed_frame = processor.process_nexrad_file(recent_files[0], site_id)
                    if processed_frame is not None:
                        site_data[site_id] = processed_frame
                        logger.info(f"Added {site_id} to mosaic")
                
            except Exception as e:
                logger.warning(f"Failed to process {site_id} for mosaic: {e}")
                continue
        
        if not site_data:
            raise HTTPException(
                status_code=404,
                detail="No radar data available for mosaic generation"
            )
        
        # Generate composite mosaic
        image_data = mosaic_svc.create_composite_mosaic(site_data)
        
        return Response(
            content=image_data,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=radar_mosaic.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mosaic generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Mosaic generation failed: {str(e)}"
        )


@router.get("/mosaic/info")
async def get_mosaic_info(
    mosaic_svc: RadarMosaicService = Depends(get_mosaic_service)
):
    """
    Get information about radar mosaic service capabilities.
    """
    try:
        return mosaic_svc.get_service_info()
        
    except Exception as e:
        logger.error(f"Failed to get mosaic info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service info: {str(e)}"
        )


# ===============================
# New Raw Radar Data Endpoints
# ===============================

@router.get("/radar-data/multi-site", response_model=MultiSiteRadarResponse)
async def get_multi_site_radar_data(
    site_ids: str,  # Comma-separated site IDs
    hours_back: int = 6,
    max_frames_per_site: int = 10,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Get raw radar data from multiple sites for composite visualizations.
    """
    start_time = time.time()
    
    try:
        # Parse site IDs
        site_list = [site.strip().upper() for site in site_ids.split(',')]
        
        # Validate sites
        invalid_sites = [site for site in site_list if site not in nexrad_svc.RADAR_SITES]
        if invalid_sites:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sites: {invalid_sites}. Available: {list(nexrad_svc.RADAR_SITES.keys())}"
            )
        
        logger.info(f"Processing multi-site request for: {site_list}")
        
        site_data = {}
        successful_sites = 0
        failed_sites = 0
        
        # Process each site
        for site_id in site_list:
            try:
                # Use the existing single-site endpoint logic
                site_response = await get_raw_radar_data(
                    site_id=site_id,
                    hours_back=hours_back,
                    max_frames=max_frames_per_site,
                    include_processing_metadata=False,
                    nexrad_svc=nexrad_svc,
                    processor=processor
                )
                site_data[site_id] = site_response
                successful_sites += 1
                
            except Exception as e:
                logger.error(f"Failed to process site {site_id}: {str(e)}")
                site_data[site_id] = RawRadarDataResponse(
                    success=False,
                    site_info=RadarSiteInfo(
                        site_id=site_id,
                        **nexrad_svc.RADAR_SITES[site_id]
                    ),
                    frames=[],
                    total_frames=0,
                    time_range={"start": datetime.now(), "end": datetime.now()},
                    processing_time_ms=0.0,
                    request_timestamp=datetime.now()
                )
                failed_sites += 1
        
        # Calculate composite bounds if we have successful sites
        composite_bounds = None
        if successful_sites > 0:
            try:
                site_locations = {}
                for site_id in site_list:
                    if site_data[site_id].success:
                        lat, lon = get_site_coordinates(site_id)
                        site_locations[site_id] = (lat, lon)
                
                if site_locations:
                    from app.utils.coordinate_utils import RadarCoordinateCalculator
                    calc = RadarCoordinateCalculator()
                    composite_dict = calc.calculate_composite_bounds(site_locations)
                    composite_bounds = CoordinateMetadata(**composite_dict)
            except Exception as e:
                logger.warning(f"Failed to calculate composite bounds: {e}")
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Multi-site request completed: {successful_sites}/{len(site_list)} successful, {processing_time_ms:.2f}ms")
        
        return MultiSiteRadarResponse(
            success=successful_sites > 0,
            site_data=site_data,
            successful_sites=successful_sites,
            failed_sites=failed_sites,
            total_sites=len(site_list),
            composite_bounds=composite_bounds,
            processing_time_ms=processing_time_ms,
            request_timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-site request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get multi-site radar data: {str(e)}"
        )


@router.get("/radar-data/{site_id}", response_model=RawRadarDataResponse)
async def get_raw_radar_data(
    site_id: str,
    hours_back: int = 6,
    max_frames: int = 20,
    include_processing_metadata: bool = False,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Get raw radar data arrays with geographic coordinates for frontend rendering.
    
    Returns processed radar data as arrays with coordinate metadata instead of images,
    perfect for Next.js frontend map integration.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing raw radar data request for site {site_id}")
        
        # Validate site
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
        recent_files = nexrad_svc.get_available_files(
            site_id, 
            hours_back=hours_back,
            max_files=max_frames
        )
        
        # Download files if needed
        if not recent_files:
            logger.info(f"No local files for {site_id}, downloading recent data")
            download_result = nexrad_svc.download_latest_files(site_id, max_files=max_frames)
            recent_files = download_result.get("file_list", [])
        
        if not recent_files:
            raise HTTPException(
                status_code=404,
                detail=f"No recent radar data available for site {site_id}"
            )
        
        # Limit to requested number of frames (respect memory limits on Render)
        effective_max_frames = min(max_frames, RADAR_MAX_BATCH_SIZE)
        if IS_RENDER and max_frames > effective_max_frames:
            logger.warning(f"Limiting frames from {max_frames} to {effective_max_frames} for memory constraints")
        recent_files = recent_files[:effective_max_frames]
        
        # Process files and create frames
        frames = []
        processing_errors = []
        cache_hits = 0
        cache_misses = 0
        
        for file_path in recent_files:
            try:
                # Process the radar file
                processed_data = processor.process_nexrad_file(
                    file_path, 
                    site_id=site_id,
                    use_cache=True
                )
                
                if processed_data is not None:
                    # Extract timestamp from filename
                    filename = os.path.basename(file_path)
                    # Format: KAMX20250802_160159_V06.ar2v
                    if len(filename) >= 19:
                        timestamp_str = filename[4:17]  # 20250802_160159
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        except ValueError:
                            timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    else:
                        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Calculate coordinate metadata
                    coordinates_dict = create_coordinate_metadata(site_id, processed_data.shape)
                    coordinates = CoordinateMetadata(**coordinates_dict)
                    
                    # Calculate intensity range
                    intensity_range = [float(processed_data.min()), float(processed_data.max())]
                    
                    # Assess data quality
                    non_zero_pixels = np.count_nonzero(processed_data)
                    total_pixels = processed_data.size
                    coverage_ratio = non_zero_pixels / total_pixels
                    
                    if coverage_ratio > 0.1:
                        data_quality = "good"
                    elif coverage_ratio > 0.05:
                        data_quality = "fair"
                    else:
                        data_quality = "poor"
                    
                    # Create processing metadata if requested
                    proc_metadata = None
                    if include_processing_metadata:
                        proc_metadata = {
                            "file_path": file_path,
                            "file_size_bytes": os.path.getsize(file_path),
                            "coverage_ratio": coverage_ratio,
                            "non_zero_pixels": int(non_zero_pixels),
                            "data_shape": list(processed_data.shape)
                        }
                    
                    # Create frame
                    frame = RadarDataFrame(
                        timestamp=timestamp,
                        data=processed_data.tolist(),
                        coordinates=coordinates,
                        intensity_range=intensity_range,
                        data_quality=data_quality,
                        processing_metadata=proc_metadata
                    )
                    frames.append(frame)
                    cache_hits += 1  # Simplified - would need cache service integration for accurate count
                else:
                    processing_errors.append(f"Failed to process: {os.path.basename(file_path)}")
                    cache_misses += 1
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                processing_errors.append(f"Error in {os.path.basename(file_path)}: {str(e)}")
        
        if not frames:
            raise HTTPException(
                status_code=500,
                detail="No radar files could be processed successfully"
            )
        
        # Sort frames by timestamp
        frames.sort(key=lambda x: x.timestamp)
        
        # Calculate time range
        time_range = {
            "start": frames[0].timestamp,
            "end": frames[-1].timestamp
        }
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create cache performance metadata
        cache_performance = {
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "processing_errors": processing_errors
        }
        
        logger.info(f"Raw radar data request completed for {site_id}: {len(frames)} frames in {processing_time_ms:.2f}ms")
        
        return RawRadarDataResponse(
            success=True,
            site_info=site_info,
            frames=frames,
            total_frames=len(frames),
            time_range=time_range,
            processing_time_ms=processing_time_ms,
            cache_performance=cache_performance,
            request_timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Raw radar data request failed for {site_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get raw radar data: {str(e)}"
        )


@router.get("/radar-timeseries/{site_id}", response_model=TimeSeriesRadarResponse)
async def get_radar_timeseries(
    site_id: str,
    start_time: str,
    end_time: str,
    max_frames: int = 50,
    include_processing_metadata: bool = False,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Get time-series radar data for a specific time range.
    
    Perfect for historical analysis and temporal visualizations.
    """
    request_start_time = time.time()
    
    try:
        logger.info(f"Processing time-series request for site {site_id}: {start_time} to {end_time}")
        
        # Validate site
        if site_id not in nexrad_svc.RADAR_SITES:
            raise HTTPException(
                status_code=404,
                detail=f"Site {site_id} not supported. Available sites: {list(nexrad_svc.RADAR_SITES.keys())}"
            )
        
        # Parse time strings
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time format. Use ISO format (YYYY-MM-DDTHH:MM:SS): {str(e)}"
            )
        
        if start_dt >= end_dt:
            raise HTTPException(
                status_code=400,
                detail="Start time must be before end time"
            )
        
        # Calculate hours back from current time to start time
        now = datetime.now()
        hours_back = int((now - start_dt).total_seconds() / 3600) + 1
        
        # Get site information
        site_info = RadarSiteInfo(
            site_id=site_id,
            **nexrad_svc.RADAR_SITES[site_id]
        )
        
        # Get available files in the broader time range
        all_files = nexrad_svc.get_available_files(
            site_id, 
            hours_back=hours_back,
            max_files=None  # Get all available files
        )
        
        # Filter files by timestamp
        filtered_files = []
        for file_path in all_files:
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            if len(filename) >= 19:
                timestamp_str = filename[4:17]  # 20250802_160159
                try:
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if start_dt <= file_timestamp <= end_dt:
                        filtered_files.append(file_path)
                except ValueError:
                    # If we can't parse the timestamp, check file modification time
                    file_timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if start_dt <= file_timestamp <= end_dt:
                        filtered_files.append(file_path)
        
        # Limit to max_frames
        if len(filtered_files) > max_frames:
            # Take evenly distributed frames across the time range
            step = len(filtered_files) // max_frames
            filtered_files = filtered_files[::step][:max_frames]
        
        if not filtered_files:
            # Try to download data for the requested time range if no local files
            total_hours = int((end_dt - start_dt).total_seconds() / 3600) + 1
            if total_hours <= 48:  # Only download if reasonable time range
                logger.info(f"No local files for time range, attempting download for {site_id}")
                nexrad_svc.download_recent_data(site_id, hours_back=total_hours)
                # Retry file search
                all_files = nexrad_svc.get_available_files(site_id, hours_back=total_hours)
                for file_path in all_files:
                    filename = os.path.basename(file_path)
                    if len(filename) >= 19:
                        timestamp_str = filename[4:17]
                        try:
                            file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            if start_dt <= file_timestamp <= end_dt:
                                filtered_files.append(file_path)
                        except ValueError:
                            continue
        
        if not filtered_files:
            raise HTTPException(
                status_code=404,
                detail=f"No radar data available for site {site_id} in time range {start_time} to {end_time}"
            )
        
        # Process files using the same logic as get_raw_radar_data
        frames = []
        for file_path in sorted(filtered_files):  # Sort by filename (chronological)
            try:
                processed_data = processor.process_nexrad_file(file_path, site_id=site_id, use_cache=True)
                
                if processed_data is not None:
                    # Extract timestamp
                    filename = os.path.basename(file_path)
                    if len(filename) >= 19:
                        timestamp_str = filename[4:17]
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        except ValueError:
                            timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    else:
                        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Create coordinate metadata
                    coordinates_dict = create_coordinate_metadata(site_id, processed_data.shape)
                    coordinates = CoordinateMetadata(**coordinates_dict)
                    
                    # Calculate frame data
                    intensity_range = [float(processed_data.min()), float(processed_data.max())]
                    
                    non_zero_pixels = np.count_nonzero(processed_data)
                    coverage_ratio = non_zero_pixels / processed_data.size
                    data_quality = "good" if coverage_ratio > 0.1 else "fair" if coverage_ratio > 0.05 else "poor"
                    
                    proc_metadata = None
                    if include_processing_metadata:
                        proc_metadata = {
                            "file_path": file_path,
                            "coverage_ratio": coverage_ratio,
                            "timestamp_source": "filename" if len(filename) >= 19 else "file_modification"
                        }
                    
                    frame = RadarDataFrame(
                        timestamp=timestamp,
                        data=processed_data.tolist(),
                        coordinates=coordinates,
                        intensity_range=intensity_range,
                        data_quality=data_quality,
                        processing_metadata=proc_metadata
                    )
                    frames.append(frame)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        if not frames:
            raise HTTPException(
                status_code=500,
                detail="No radar files in the time range could be processed"
            )
        
        # Sort frames by timestamp
        frames.sort(key=lambda x: x.timestamp)
        
        # Calculate metrics
        actual_start = frames[0].timestamp
        actual_end = frames[-1].timestamp
        
        # Calculate temporal resolution
        if len(frames) > 1:
            time_diffs = []
            for i in range(1, len(frames)):
                diff_seconds = (frames[i].timestamp - frames[i-1].timestamp).total_seconds()
                time_diffs.append(diff_seconds / 60)  # Convert to minutes
            temporal_resolution_minutes = sum(time_diffs) / len(time_diffs)
        else:
            temporal_resolution_minutes = 0.0
        
        # Calculate data completeness
        requested_duration = (end_dt - start_dt).total_seconds()
        actual_duration = (actual_end - actual_start).total_seconds()
        data_completeness = min(1.0, actual_duration / requested_duration) if requested_duration > 0 else 1.0
        
        processing_time_ms = (time.time() - request_start_time) * 1000
        
        logger.info(f"Time-series request completed for {site_id}: {len(frames)} frames, "
                   f"completeness: {data_completeness:.2%}, {processing_time_ms:.2f}ms")
        
        return TimeSeriesRadarResponse(
            success=True,
            site_info=site_info,
            frames=frames,
            total_frames=len(frames),
            time_range={"start": start_dt, "end": end_dt},
            actual_time_range={"start": actual_start, "end": actual_end},
            temporal_resolution_minutes=temporal_resolution_minutes,
            data_completeness=data_completeness,
            processing_time_ms=processing_time_ms,
            request_timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Time-series request failed for {site_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get time-series radar data: {str(e)}"
        )


@router.get("/radar-frame/{site_id}", response_model=RadarDataFrame)
async def get_single_radar_frame(
    site_id: str,
    timestamp: str = None,
    include_processing_metadata: bool = False,
    nexrad_svc: NEXRADDataService = Depends(get_nexrad_service),
    processor: RadarProcessingService = Depends(get_radar_processor)
):
    """
    Get a single radar frame, either the latest or for a specific timestamp.
    """
    try:
        # Validate site
        if site_id not in nexrad_svc.RADAR_SITES:
            raise HTTPException(
                status_code=404,
                detail=f"Site {site_id} not supported. Available sites: {list(nexrad_svc.RADAR_SITES.keys())}"
            )
        
        # Get available files
        recent_files = nexrad_svc.get_available_files(site_id, hours_back=6, max_files=10)
        
        if not recent_files:
            # Download latest files if none available
            download_result = nexrad_svc.download_latest_files(site_id, max_files=5)
            recent_files = download_result.get("file_list", [])
        
        if not recent_files:
            raise HTTPException(
                status_code=404,
                detail=f"No radar data available for site {site_id}"
            )
        
        # Find the requested file
        target_file = None
        if timestamp:
            # Parse requested timestamp
            try:
                target_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid timestamp format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
                )
            
            # Find closest file to requested timestamp
            best_diff = float('inf')
            for file_path in recent_files:
                filename = os.path.basename(file_path)
                if len(filename) >= 19:
                    timestamp_str = filename[4:17]
                    try:
                        file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        diff = abs((file_timestamp - target_dt).total_seconds())
                        if diff < best_diff:
                            best_diff = diff
                            target_file = file_path
                    except ValueError:
                        continue
        else:
            # Use most recent file
            target_file = recent_files[0]
        
        if not target_file:
            raise HTTPException(
                status_code=404,
                detail=f"No radar data found for requested timestamp"
            )
        
        # Process the file
        processed_data = processor.process_nexrad_file(target_file, site_id=site_id, use_cache=True)
        
        if processed_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to process radar file"
            )
        
        # Extract timestamp from filename
        filename = os.path.basename(target_file)
        if len(filename) >= 19:
            timestamp_str = filename[4:17]
            try:
                frame_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                frame_timestamp = datetime.fromtimestamp(os.path.getmtime(target_file))
        else:
            frame_timestamp = datetime.fromtimestamp(os.path.getmtime(target_file))
        
        # Create coordinate metadata
        coordinates_dict = create_coordinate_metadata(site_id, processed_data.shape)
        coordinates = CoordinateMetadata(**coordinates_dict)
        
        # Calculate frame data
        intensity_range = [float(processed_data.min()), float(processed_data.max())]
        
        non_zero_pixels = np.count_nonzero(processed_data)
        coverage_ratio = non_zero_pixels / processed_data.size
        data_quality = "good" if coverage_ratio > 0.1 else "fair" if coverage_ratio > 0.05 else "poor"
        
        proc_metadata = None
        if include_processing_metadata:
            proc_metadata = {
                "file_path": target_file,
                "file_size_bytes": os.path.getsize(target_file),
                "coverage_ratio": coverage_ratio,
                "requested_timestamp": timestamp,
                "timestamp_source": "filename" if len(filename) >= 19 else "file_modification"
            }
        
        return RadarDataFrame(
            timestamp=frame_timestamp,
            data=processed_data.tolist(),
            coordinates=coordinates,
            intensity_range=intensity_range,
            data_quality=data_quality,
            processing_metadata=proc_metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single frame request failed for {site_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get radar frame: {str(e)}"
        )


# Note: Exception handlers are registered at the app level in main.py
# Individual endpoints handle their own errors with try/catch blocks