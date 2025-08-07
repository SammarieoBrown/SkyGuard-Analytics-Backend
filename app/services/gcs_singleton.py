"""
Singleton GCS Storage Service to avoid re-initialization.
"""
import logging
from typing import Optional
from app.config import USE_GCS_STORAGE, GCS_BUCKET_NAME, GCS_CREDENTIALS
from app.services.gcs_storage_service import GCSStorageService

logger = logging.getLogger(__name__)

# Global singleton instance
_gcs_service_instance: Optional[GCSStorageService] = None


def get_gcs_service() -> Optional[GCSStorageService]:
    """
    Get or create the singleton GCS service instance.
    
    Returns:
        GCSStorageService instance or None if GCS is disabled/fails
    """
    global _gcs_service_instance
    
    if not USE_GCS_STORAGE:
        return None
    
    if _gcs_service_instance is None:
        try:
            _gcs_service_instance = GCSStorageService(GCS_BUCKET_NAME, GCS_CREDENTIALS)
            logger.info(f"GCS Service singleton initialized with bucket: {GCS_BUCKET_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS service singleton: {e}")
            return None
    
    return _gcs_service_instance


def reset_gcs_service():
    """Reset the singleton instance (mainly for testing)."""
    global _gcs_service_instance
    _gcs_service_instance = None