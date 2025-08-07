"""
Google Cloud Storage Service for managing NEXRAD radar data files.
Provides methods for uploading, downloading, and managing files in GCS.
"""
import os
import io
import json
import logging
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime, timedelta
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class GCSStorageService:
    """
    Service for managing NEXRAD radar data in Google Cloud Storage.
    """
    
    def __init__(self, bucket_name: str, credentials_json: Optional[str] = None):
        """
        Initialize GCS storage service.
        
        Args:
            bucket_name: Name of the GCS bucket
            credentials_json: JSON string containing service account credentials
        """
        self.bucket_name = bucket_name
        
        try:
            if credentials_json:
                # Parse credentials from JSON string
                credentials_dict = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_dict
                )
                self.client = storage.Client(credentials=credentials)
            else:
                # Use default credentials (for local development with gcloud auth)
                self.client = storage.Client()
            
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"GCS Storage Service initialized with bucket: {bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def upload_file(self, file_data: bytes, blob_name: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload file data to GCS bucket.
        
        Args:
            file_data: File content as bytes
            blob_name: Path/name for the file in GCS (e.g., 'nexrad/KAMX/2024-01-01/file.bin')
            metadata: Optional metadata to attach to the file
            
        Returns:
            bool: True if upload successful
        """
        try:
            blob = self.bucket.blob(blob_name)
            
            # Set metadata if provided
            if metadata:
                blob.metadata = metadata
            
            # Upload the file
            blob.upload_from_string(file_data, content_type='application/octet-stream')
            
            logger.debug(f"Uploaded {blob_name} to GCS ({len(file_data)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {blob_name} to GCS: {e}")
            return False
    
    def download_file(self, blob_name: str) -> Optional[bytes]:
        """
        Download file from GCS bucket.
        
        Args:
            blob_name: Path/name of the file in GCS
            
        Returns:
            bytes: File content, or None if not found
        """
        try:
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                logger.debug(f"File {blob_name} not found in GCS")
                return None
            
            file_data = blob.download_as_bytes()
            logger.debug(f"Downloaded {blob_name} from GCS ({len(file_data)} bytes)")
            return file_data
            
        except Exception as e:
            logger.error(f"Failed to download {blob_name} from GCS: {e}")
            return None
    
    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a file exists in GCS.
        
        Args:
            blob_name: Path/name of the file in GCS
            
        Returns:
            bool: True if file exists
        """
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check existence of {blob_name}: {e}")
            return False
    
    def list_files(self, prefix: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List files in GCS with a given prefix.
        
        Args:
            prefix: Prefix to filter files (e.g., 'nexrad/KAMX/')
            max_results: Maximum number of results to return
            
        Returns:
            List of file info dictionaries
        """
        try:
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
            
            files = []
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created,
                    'updated': blob.updated,
                    'metadata': blob.metadata
                })
            
            logger.debug(f"Listed {len(files)} files with prefix '{prefix}'")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []
    
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a file from GCS.
        
        Args:
            blob_name: Path/name of the file in GCS
            
        Returns:
            bool: True if deletion successful
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.debug(f"Deleted {blob_name} from GCS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {blob_name} from GCS: {e}")
            return False
    
    def delete_old_files(self, prefix: str, days_old: int) -> int:
        """
        Delete files older than specified days.
        
        Args:
            prefix: Prefix to filter files (e.g., 'nexrad/')
            days_old: Delete files older than this many days
            
        Returns:
            int: Number of files deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            deleted_count = 0
            for blob in blobs:
                if blob.time_created < cutoff_date:
                    blob.delete()
                    deleted_count += 1
                    logger.debug(f"Deleted old file: {blob.name}")
            
            logger.info(f"Deleted {deleted_count} files older than {days_old} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete old files: {e}")
            return 0
    
    def get_blob_url(self, blob_name: str, expiration_hours: int = 1) -> Optional[str]:
        """
        Generate a signed URL for direct access to a file.
        
        Args:
            blob_name: Path/name of the file in GCS
            expiration_hours: Hours until the URL expires
            
        Returns:
            str: Signed URL, or None if error
        """
        try:
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                return None
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=expiration_hours),
                method="GET"
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {blob_name}: {e}")
            return None
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage usage information for the bucket.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            total_size = 0
            file_count = 0
            
            for blob in self.bucket.list_blobs():
                total_size += blob.size
                file_count += 1
            
            return {
                'bucket_name': self.bucket_name,
                'total_files': file_count,
                'total_size_mb': total_size / (1024 * 1024),
                'total_size_gb': total_size / (1024 * 1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {
                'bucket_name': self.bucket_name,
                'error': str(e)
            }
    
    def create_nexrad_blob_name(self, site_id: str, date: datetime, filename: str) -> str:
        """
        Create a standardized blob name for NEXRAD files.
        
        Args:
            site_id: Radar site identifier (e.g., 'KAMX')
            date: Date of the radar data
            filename: Original filename
            
        Returns:
            str: Blob name like 'nexrad/KAMX/2024-01-01/filename'
        """
        date_str = date.strftime('%Y-%m-%d')
        return f"nexrad/{site_id}/{date_str}/{filename}"