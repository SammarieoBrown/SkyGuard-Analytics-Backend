"""
Data Pipeline Background Tasks - Automated tasks for NEXRAD data management.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from app.services.nexrad_data_service import NEXRADDataService
from app.config import NEXRAD_DATA_RETENTION_DAYS

logger = logging.getLogger(__name__)


class DataPipelineScheduler:
    """
    Background task scheduler for NEXRAD data pipeline operations.
    Handles automated data fetching and cleanup.
    """
    
    def __init__(self):
        """Initialize the data pipeline scheduler."""
        self.nexrad_service = NEXRADDataService()
        self.is_running = False
        
        logger.info("DataPipelineScheduler initialized")
    
    async def fetch_latest_data(self, hours_back: int = 6) -> Dict[str, Any]:
        """
        Background task to fetch latest radar data for all sites.
        
        Args:
            hours_back: How many hours back to fetch data
            
        Returns:
            dict: Results of the data fetch operation
        """
        logger.info(f"Starting automated data fetch ({hours_back} hours back)")
        
        try:
            results = self.nexrad_service.download_all_sites_recent(hours_back)
            
            # Log summary
            total_files = sum(site_result.get("total_files", 0) for site_result in results.values())
            successful_sites = sum(1 for site_result in results.values() 
                                 if site_result.get("status") != "failed")
            
            logger.info(f"Automated data fetch complete: "
                       f"{successful_sites}/{len(results)} sites successful, "
                       f"{total_files} files downloaded")
            
            return {
                "status": "completed",
                "timestamp": datetime.now(),
                "sites_processed": len(results),
                "successful_sites": successful_sites,
                "total_files": total_files,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Automated data fetch failed: {str(e)}")
            return {
                "status": "failed",
                "timestamp": datetime.now(),
                "error": str(e)
            }
    
    async def cleanup_old_data(self, days_to_keep: int = None) -> Dict[str, Any]:
        """
        Background task to clean up old radar data files.
        
        Args:
            days_to_keep: Number of days to retain (default from config)
            
        Returns:
            dict: Results of the cleanup operation
        """
        if days_to_keep is None:
            days_to_keep = NEXRAD_DATA_RETENTION_DAYS
        
        logger.info(f"Starting automated data cleanup (keeping {days_to_keep} days)")
        
        try:
            cleanup_stats = self.nexrad_service.cleanup_old_data(days_to_keep)
            
            logger.info(f"Automated cleanup complete: "
                       f"{cleanup_stats['files_removed']} files removed, "
                       f"{cleanup_stats['bytes_freed'] / 1024 / 1024:.1f} MB freed")
            
            return {
                "status": "completed",
                "timestamp": datetime.now(),
                "days_kept": days_to_keep,
                "cleanup_stats": cleanup_stats
            }
            
        except Exception as e:
            logger.error(f"Automated cleanup failed: {str(e)}")
            return {
                "status": "failed",
                "timestamp": datetime.now(),
                "error": str(e)
            }
    
    async def health_check_pipeline(self) -> Dict[str, Any]:
        """
        Background task to check health of data pipeline.
        
        Returns:
            dict: Health check results
        """
        logger.info("Running data pipeline health check")
        
        try:
            service_status = self.nexrad_service.get_service_status()
            
            # Check data freshness for each site
            health_results = {
                "status": "healthy",
                "timestamp": datetime.now(),
                "sites": {},
                "overall_health": True
            }
            
            for site_id in self.nexrad_service.RADAR_SITES.keys():
                recent_files = self.nexrad_service.get_available_files(site_id, hours_back=24)
                
                site_health = {
                    "files_available": len(recent_files),
                    "data_age_hours": None,
                    "status": "unknown"
                }
                
                if recent_files:
                    # Check age of most recent file
                    import os
                    latest_file = recent_files[0]
                    file_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
                    age_hours = (datetime.now() - file_time).total_seconds() / 3600
                    
                    site_health["data_age_hours"] = age_hours
                    
                    if age_hours < 2:
                        site_health["status"] = "excellent"
                    elif age_hours < 6:
                        site_health["status"] = "good"
                    elif age_hours < 24:
                        site_health["status"] = "fair"
                    else:
                        site_health["status"] = "poor"
                        health_results["overall_health"] = False
                else:
                    site_health["status"] = "no_data"
                    health_results["overall_health"] = False
                
                health_results["sites"][site_id] = site_health
            
            if not health_results["overall_health"]:
                health_results["status"] = "degraded"
            
            logger.info(f"Health check complete: {health_results['status']}")
            
            return health_results
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error",
                "timestamp": datetime.now(),
                "error": str(e)
            }
    
    async def run_scheduled_tasks(self):
        """
        Main scheduler loop for running automated tasks.
        This would typically be called from a background worker or cron job.
        """
        logger.info("Starting data pipeline scheduler")
        self.is_running = True
        
        last_fetch = datetime.now() - timedelta(hours=1)  # Force initial fetch
        last_cleanup = datetime.now() - timedelta(days=1)  # Force initial cleanup check
        last_health_check = datetime.now() - timedelta(hours=1)
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Fetch latest data every hour
                if (current_time - last_fetch).total_seconds() >= 3600:  # 1 hour
                    await self.fetch_latest_data(hours_back=2)
                    last_fetch = current_time
                
                # Run cleanup daily
                if (current_time - last_cleanup).total_seconds() >= 86400:  # 24 hours
                    await self.cleanup_old_data()
                    last_cleanup = current_time
                
                # Health check every 30 minutes
                if (current_time - last_health_check).total_seconds() >= 1800:  # 30 minutes
                    health_result = await self.health_check_pipeline()
                    last_health_check = current_time
                    
                    # Log warnings if health is degraded
                    if health_result.get("status") != "healthy":
                        logger.warning(f"Data pipeline health check warning: {health_result}")
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)  # Sleep for 1 minute on error
    
    def stop_scheduler(self):
        """Stop the scheduler loop."""
        logger.info("Stopping data pipeline scheduler")
        self.is_running = False


# Global scheduler instance
pipeline_scheduler = DataPipelineScheduler()


# Individual task functions for manual execution
async def manual_data_fetch(hours_back: int = 6) -> Dict[str, Any]:
    """
    Manually trigger data fetch task.
    
    Args:
        hours_back: Hours of data to fetch
        
    Returns:
        dict: Task results
    """
    return await pipeline_scheduler.fetch_latest_data(hours_back)


async def manual_cleanup(days_to_keep: int = None) -> Dict[str, Any]:
    """
    Manually trigger cleanup task.
    
    Args:
        days_to_keep: Days of data to retain
        
    Returns:
        dict: Task results
    """
    return await pipeline_scheduler.cleanup_old_data(days_to_keep)


async def manual_health_check() -> Dict[str, Any]:
    """
    Manually trigger health check task.
    
    Returns:
        dict: Health check results
    """
    return await pipeline_scheduler.health_check_pipeline()


def start_background_scheduler():
    """
    Start the background scheduler.
    This should be called during application startup.
    """
    import threading
    
    def run_scheduler():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(pipeline_scheduler.run_scheduled_tasks())
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    logger.info("Background data pipeline scheduler started")


def stop_background_scheduler():
    """
    Stop the background scheduler.
    This should be called during application shutdown.
    """
    pipeline_scheduler.stop_scheduler()
    logger.info("Background data pipeline scheduler stopped")