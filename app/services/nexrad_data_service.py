"""
NEXRAD Data Service - Clean production implementation for fetching NEXRAD Level-II data.
Fetches real-time radar data from AWS S3 public storage and stores in GCS.
"""
import os
import requests
import tempfile
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import xml.etree.ElementTree as ET
from app.config import USE_GCS_STORAGE, GCS_BUCKET_NAME, GCS_CREDENTIALS

logger = logging.getLogger(__name__)


class NEXRADDataService:
    """
    Production service for fetching NEXRAD Level-II radar data.
    Focuses on KAMX and KATX sites for weather nowcasting.
    """
    
    # Radar sites for weather nowcasting (trained model sites)
    RADAR_SITES = {
        "KAMX": {
            "name": "Miami",
            "location": "Miami, FL",
            "coordinates": [25.6112, -80.4128],
            "description": "Atlantic hurricanes, tropical storms"
        },
        "KATX": {
            "name": "Seattle", 
            "location": "Seattle, WA",
            "coordinates": [48.1947, -122.4956],
            "description": "Pacific Northwest storms, mountain effects"
        }
    }
    
    def __init__(self, data_dir: str = None, max_workers: int = 4):
        """
        Initialize NEXRAD data service.
        
        Args:
            data_dir: Directory to store downloaded radar data (used if GCS is disabled)
            max_workers: Maximum number of concurrent download threads
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        self.max_workers = max_workers
        # AWS S3 NEXRAD Level-II bucket
        self.base_url = "https://noaa-nexrad-level2.s3.amazonaws.com"
        self.lock = threading.Lock()
        
        # Initialize GCS if enabled
        self.gcs_service = None
        if USE_GCS_STORAGE:
            try:
                from app.services.gcs_storage_service import GCSStorageService
                self.gcs_service = GCSStorageService(GCS_BUCKET_NAME, GCS_CREDENTIALS)
                logger.info(f"NEXRADDataService initialized with GCS storage (bucket: {GCS_BUCKET_NAME})")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS, falling back to local storage: {e}")
                self.gcs_service = None
        
        # Ensure local data directory exists (for fallback or if GCS disabled)
        if not USE_GCS_STORAGE or not self.gcs_service:
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"NEXRADDataService initialized with local storage, data_dir: {self.data_dir}")
    
    def _get_default_data_dir(self) -> str:
        """Get default data directory path."""
        base_dir = Path(__file__).parent.parent.parent
        return str(base_dir / "data" / "radar")
    
    def _log(self, site_id: str, message: str):
        """Thread-safe logging with site context."""
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            logger.info(f"[{timestamp}] {site_id}: {message}")
    
    def _create_site_directory(self, site_id: str, date: datetime) -> str:
        """Create directory structure for site and date."""
        date_str = date.strftime('%Y-%m-%d')
        site_dir = os.path.join(self.data_dir, site_id, date_str)
        os.makedirs(site_dir, exist_ok=True)
        return site_dir
    
    def _list_available_files(self, site_id: str, date: datetime) -> List[str]:
        """
        List available NEXRAD files for a specific site and date from AWS S3.
        
        Args:
            site_id: Radar site identifier (e.g., 'KAMX')
            date: Date for the data
            
        Returns:
            list: List of available file keys
        """
        # Construct the S3 listing URL
        prefix = f"{date.year:04d}/{date.month:02d}/{date.day:02d}/{site_id}/"
        list_url = f"{self.base_url}/?prefix={prefix}"
        
        try:
            response = requests.get(list_url, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
            
            files = []
            for content in root.findall('.//s3:Contents', namespace):
                key = content.find('s3:Key', namespace)
                if key is not None and key.text:
                    # Filter for actual data files (ending with _V06)
                    if key.text.endswith('_V06') or key.text.endswith('_V06.gz'):
                        files.append(key.text)
            
            return sorted(files)
            
        except Exception as e:
            logger.debug(f"Could not list files for {site_id} on {date}: {e}")
            return []
    
    def _build_file_url(self, file_key: str) -> str:
        """
        Build full URL for a specific file.
        
        Args:
            file_key: S3 key for the file
            
        Returns:
            str: Complete URL for the file
        """
        return f"{self.base_url}/{file_key}"
    
    def _download_files_for_hour(self, site_id: str, date: datetime, hour: int) -> Tuple[int, List[str]]:
        """
        Download NEXRAD files for a specific hour from AWS S3.
        
        Args:
            site_id: Radar site identifier
            date: Date for the data
            hour: Hour to download (0-23)
            
        Returns:
            tuple: (number_of_files, list_of_filenames)
        """
        try:
            site_dir = self._create_site_directory(site_id, date)
            
            # List available files for this date
            available_files = self._list_available_files(site_id, date)
            
            # Filter files for the specific hour
            hour_str = f"{hour:02d}"
            hour_files = [f for f in available_files if f"_{hour_str}" in f]
            
            if not hour_files:
                return 0, []
            
            # Check if we already have files from this hour
            hour_pattern = f"{site_id}{date.strftime('%Y%m%d')}_{hour:02d}"
            existing_files = [f for f in os.listdir(site_dir) if f.startswith(hour_pattern)]
            if len(existing_files) >= len(hour_files):
                return len(existing_files), existing_files  # Already downloaded
            
            # Download individual files
            downloaded_files = []
            for file_key in hour_files[:6]:  # Limit to 6 files per hour (every ~10 minutes)
                filename = os.path.basename(file_key)
                
                # Check if file already exists (GCS or local)
                if self.gcs_service:
                    blob_name = self.gcs_service.create_nexrad_blob_name(site_id, date, filename)
                    if self.gcs_service.file_exists(blob_name):
                        downloaded_files.append(filename)
                        continue
                else:
                    file_path = os.path.join(site_dir, filename)
                    if os.path.exists(file_path):
                        downloaded_files.append(filename)
                        continue
                
                # Download file
                file_url = self._build_file_url(file_key)
                try:
                    response = requests.get(file_url, timeout=60, stream=True)
                    response.raise_for_status()
                    
                    # Collect file data in memory
                    file_data = b''
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file_data += chunk
                    
                    # Save to GCS or local
                    if self.gcs_service:
                        blob_name = self.gcs_service.create_nexrad_blob_name(site_id, date, filename)
                        metadata = {
                            'site_id': site_id,
                            'date': date.strftime('%Y-%m-%d'),
                            'hour': str(hour),
                            'source': 'AWS_S3'
                        }
                        if self.gcs_service.upload_file(file_data, blob_name, metadata):
                            downloaded_files.append(filename)
                    else:
                        # Fallback to local storage
                        file_path = os.path.join(site_dir, filename)
                        with open(file_path, 'wb') as f:
                            f.write(file_data)
                        downloaded_files.append(filename)
                    
                except Exception as e:
                    logger.debug(f"Failed to download {filename}: {e}")
                    continue
            
            if downloaded_files:
                self._log(site_id, f"✅ Hour {hour:02d}: {len(downloaded_files)} files")
            
            return len(downloaded_files), downloaded_files
            
        except Exception as e:
            self._log(site_id, f"❌ Error processing hour {hour:02d}: {str(e)}")
            return 0, []
    
    def download_recent_data(self, site_id: str, hours_back: int = 12) -> Dict[str, any]:
        """
        Download recent radar data for a site.
        
        Args:
            site_id: Radar site identifier (KAMX or KATX)
            hours_back: How many hours back to fetch data
            
        Returns:
            dict: Download results summary
        """
        if site_id not in self.RADAR_SITES:
            raise ValueError(f"Unsupported site: {site_id}. Supported: {list(self.RADAR_SITES.keys())}")
        
        logger.info(f"Downloading recent data for {site_id} ({hours_back} hours back)")
        
        results = {
            "site_id": site_id,
            "site_info": self.RADAR_SITES[site_id],
            "hours_requested": hours_back,
            "hours_successful": 0,
            "total_files": 0,
            "files_by_hour": {},
            "start_time": datetime.now()
        }
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Download data for each hour
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            current_time = start_time
            while current_time <= end_time:
                hour = current_time.hour
                future = executor.submit(
                    self._download_files_for_hour, 
                    site_id, 
                    current_time.date() if hasattr(current_time, 'date') else current_time,
                    hour
                )
                futures[future] = (current_time, hour)
                current_time += timedelta(hours=1)
            
            # Collect results
            for future in as_completed(futures):
                time_info, hour = futures[future]
                try:
                    file_count, filenames = future.result()
                    results["total_files"] += file_count
                    if file_count > 0:
                        results["hours_successful"] += 1
                        results["files_by_hour"][hour] = {
                            "count": file_count,
                            "files": filenames
                        }
                except Exception as e:
                    logger.error(f"Failed to process hour {hour}: {e}")
        
        results["end_time"] = datetime.now()
        results["duration_seconds"] = (results["end_time"] - results["start_time"]).total_seconds()
        
        logger.info(f"Download complete for {site_id}: "
                   f"{results['hours_successful']}/{results['hours_requested']} hours, "
                   f"{results['total_files']} files")
        
        return results
    
    def download_latest_files(self, site_id: str, max_files: int = 10) -> Dict[str, Any]:
        """
        Download only the latest N files for immediate processing (optimized for speed).
        
        This method downloads individual files directly from AWS S3,
        significantly reducing download time for real-time predictions.
        
        Args:
            site_id: Radar site identifier
            max_files: Maximum number of latest files to download
            
        Returns:
            dict: Download results with timing information
        """
        if site_id not in self.RADAR_SITES:
            raise ValueError(f"Unsupported site: {site_id}")
        
        logger.info(f"Downloading latest {max_files} files for {site_id}")
        start_time = datetime.now()
        
        results = {
            "site_id": site_id,
            "max_files_requested": max_files,
            "files_downloaded": 0,
            "files_already_exist": 0,
            "download_errors": 0,
            "file_list": [],
            "start_time": start_time
        }
        
        try:
            # Check what files we already have locally
            existing_files = self.get_available_files(site_id, hours_back=6, max_files=max_files)
            
            if len(existing_files) >= max_files:
                results["files_already_exist"] = len(existing_files)
                results["file_list"] = existing_files[:max_files]
                logger.info(f"Already have {len(existing_files)} recent files for {site_id}")
                return results
            
            # Try to download from most recent dates
            downloaded_count = 0
            all_files = []
            
            # Check last 3 days
            for days_back in range(3):
                date = datetime.utcnow().date() - timedelta(days=days_back)
                site_dir = self._create_site_directory(site_id, date)
                
                # List available files for this date
                available_files = self._list_available_files(site_id, date)
                
                if available_files:
                    # Sort by timestamp (most recent first)
                    available_files.sort(reverse=True)
                    
                    # Download up to max_files
                    for file_key in available_files[:max_files - downloaded_count]:
                        filename = os.path.basename(file_key)
                        file_path = os.path.join(site_dir, filename)
                        
                        # Skip if already exists
                        if os.path.exists(file_path):
                            all_files.append(file_path)
                            continue
                        
                        # Download file
                        file_url = self._build_file_url(file_key)
                        try:
                            response = requests.get(file_url, timeout=60, stream=True)
                            response.raise_for_status()
                            
                            # Save file
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            
                            all_files.append(file_path)
                            downloaded_count += 1
                            results["files_downloaded"] += 1
                            
                            if downloaded_count >= max_files:
                                break
                                
                        except Exception as e:
                            logger.debug(f"Failed to download {filename}: {e}")
                            results["download_errors"] += 1
                
                if downloaded_count >= max_files:
                    break
            
            # Sort by modification time (most recent first)
            all_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
            results["file_list"] = all_files[:max_files]
            
        except Exception as e:
            logger.error(f"Failed to download latest files for {site_id}: {e}")
            results["download_errors"] += 1
            results["error"] = str(e)
        
        results["end_time"] = datetime.now()
        results["duration_seconds"] = (results["end_time"] - results["start_time"]).total_seconds()
        
        logger.info(f"Latest file download complete for {site_id}: "
                   f"{results['files_downloaded']} new files, "
                   f"{results['files_already_exist']} existing files, "
                   f"{results['duration_seconds']:.2f}s")
        
        return results
    
    def download_all_sites_recent(self, hours_back: int = 12) -> Dict[str, Dict]:
        """
        Download recent data for all supported radar sites.
        
        Args:
            hours_back: How many hours back to fetch data
            
        Returns:
            dict: Results for each site
        """
        logger.info(f"Downloading recent data for all sites ({hours_back} hours back)")
        
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.RADAR_SITES)) as executor:
            futures = {
                executor.submit(self.download_recent_data, site_id, hours_back): site_id
                for site_id in self.RADAR_SITES.keys()
            }
            
            for future in as_completed(futures):
                site_id = futures[future]
                try:
                    result = future.result()
                    all_results[site_id] = result
                except Exception as e:
                    logger.error(f"Failed to download data for {site_id}: {e}")
                    all_results[site_id] = {
                        "site_id": site_id,
                        "error": str(e),
                        "status": "failed"
                    }
        
        return all_results
    
    def get_available_files(self, site_id: str, hours_back: int = 24, max_files: int = None) -> List[str]:
        """
        Get list of available radar files for a site with optional limits.
        
        Args:
            site_id: Radar site identifier
            hours_back: How far back to look for files
            max_files: Maximum number of files to return (most recent first)
            
        Returns:
            list: List of available file paths (or blob names if using GCS)
        """
        if site_id not in self.RADAR_SITES:
            raise ValueError(f"Unsupported site: {site_id}")
        
        files = []
        
        if self.gcs_service:
            # Get files from GCS
            prefix = f"nexrad/{site_id}/"
            gcs_files = self.gcs_service.list_files(prefix)
            
            # Filter by time if needed
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            for file_info in gcs_files:
                if file_info['created'] and file_info['created'] >= cutoff_time:
                    files.append(file_info['name'])
            
            # Sort by creation time (most recent first)
            files.sort(reverse=True)
        else:
            # Fallback to local storage
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=hours_back // 24 + 1)
            
            current_date = start_date
            while current_date <= end_date:
                date_dir = os.path.join(self.data_dir, site_id, current_date.strftime('%Y-%m-%d'))
                
                if os.path.exists(date_dir):
                    for filename in os.listdir(date_dir):
                        # AWS S3 files end with _V06 or _V06.gz
                        if filename.endswith('_V06') or filename.endswith('_V06.gz') or filename.endswith('.ar2v'):
                            file_path = os.path.join(date_dir, filename)
                            files.append(file_path)
                
                current_date += timedelta(days=1)
            
            # Sort by modification time (most recent first)
            files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
        
        # Apply max_files limit if specified
        if max_files and len(files) > max_files:
            files = files[:max_files]
            logger.debug(f"Limited results to {max_files} most recent files for {site_id}")
        
        return files
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> Dict[str, int]:
        """
        Clean up old radar data files.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            dict: Cleanup statistics
        """
        logger.info(f"Cleaning up radar data older than {days_to_keep} days")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_stats = {
            "files_removed": 0,
            "directories_removed": 0,
            "bytes_freed": 0
        }
        
        for site_id in self.RADAR_SITES.keys():
            site_dir = os.path.join(self.data_dir, site_id)
            
            if not os.path.exists(site_dir):
                continue
            
            for date_dir_name in os.listdir(site_dir):
                try:
                    # Parse date from directory name
                    date_obj = datetime.strptime(date_dir_name, '%Y-%m-%d')
                    
                    if date_obj < cutoff_date:
                        date_dir_path = os.path.join(site_dir, date_dir_name)
                        
                        # Calculate size and remove files
                        for filename in os.listdir(date_dir_path):
                            file_path = os.path.join(date_dir_path, filename)
                            if os.path.isfile(file_path):
                                cleanup_stats["bytes_freed"] += os.path.getsize(file_path)
                                os.remove(file_path)
                                cleanup_stats["files_removed"] += 1
                        
                        # Remove directory
                        os.rmdir(date_dir_path)
                        cleanup_stats["directories_removed"] += 1
                        
                        logger.info(f"Removed old data directory: {date_dir_path}")
                        
                except ValueError:
                    # Skip directories that don't match date format
                    continue
                except Exception as e:
                    logger.error(f"Error cleaning up {date_dir_name}: {e}")
        
        logger.info(f"Cleanup complete: {cleanup_stats['files_removed']} files, "
                   f"{cleanup_stats['directories_removed']} directories, "
                   f"{cleanup_stats['bytes_freed'] / 1024 / 1024:.1f} MB freed")
        
        return cleanup_stats
    
    def get_service_status(self) -> Dict[str, any]:
        """
        Get current status of the NEXRAD data service.
        
        Returns:
            dict: Service status information
        """
        status = {
            "service": "NEXRADDataService",
            "data_dir": self.data_dir,
            "supported_sites": list(self.RADAR_SITES.keys()),
            "site_details": self.RADAR_SITES,
            "storage_info": {}
        }
        
        # Check storage for each site
        for site_id in self.RADAR_SITES.keys():
            site_dir = os.path.join(self.data_dir, site_id)
            site_info = {
                "directory_exists": os.path.exists(site_dir),
                "available_dates": [],
                "total_files": 0,
                "total_size_mb": 0
            }
            
            if os.path.exists(site_dir):
                # Get available dates
                for date_dir in os.listdir(site_dir):
                    if os.path.isdir(os.path.join(site_dir, date_dir)):
                        site_info["available_dates"].append(date_dir)
                        
                        # Count files and size
                        date_path = os.path.join(site_dir, date_dir)
                        for filename in os.listdir(date_path):
                            file_path = os.path.join(date_path, filename)
                            if os.path.isfile(file_path):
                                site_info["total_files"] += 1
                                site_info["total_size_mb"] += os.path.getsize(file_path) / 1024 / 1024
                
                site_info["available_dates"].sort(reverse=True)  # Most recent first
            
            status["storage_info"][site_id] = site_info
        
        return status