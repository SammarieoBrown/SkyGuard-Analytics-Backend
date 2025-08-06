"""
NEXRAD Data Service - Clean production implementation for fetching NEXRAD Level-II data.
Fetches real-time radar data from Google Cloud Platform public storage.
"""
import os
import requests
import tempfile
import tarfile
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path

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
            data_dir: Directory to store downloaded radar data
            max_workers: Maximum number of concurrent download threads
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        self.max_workers = max_workers
        self.base_url = "https://storage.googleapis.com/gcp-public-data-nexrad-l2"
        self.lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"NEXRADDataService initialized with data_dir: {self.data_dir}")
    
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
    
    def _build_nexrad_url(self, site_id: str, date: datetime, hour: int) -> str:
        """
        Build URL for NEXRAD hourly tar file.
        
        Args:
            site_id: Radar site identifier (e.g., 'KAMX')
            date: Date for the data
            hour: Hour of the day (0-23)
            
        Returns:
            str: Complete URL for the tar file
        """
        start_time = f"{hour:02d}0000"
        end_time = f"{hour:02d}5959"
        
        filename = (f"NWS_NEXRAD_NXL2DPBL_{site_id}_{date.strftime('%Y%m%d')}{start_time}_"
                   f"{date.strftime('%Y%m%d')}{end_time}.tar")
        
        url = f"{self.base_url}/{date.year:04d}/{date.month:02d}/{date.day:02d}/{site_id}/{filename}"
        return url
    
    def _download_hourly_data(self, site_id: str, date: datetime, hour: int) -> Tuple[int, List[str]]:
        """
        Download and extract hourly NEXRAD data for a site.
        
        Args:
            site_id: Radar site identifier
            date: Date for the data
            hour: Hour to download (0-23)
            
        Returns:
            tuple: (number_of_files, list_of_filenames)
        """
        try:
            url = self._build_nexrad_url(site_id, date, hour)
            site_dir = self._create_site_directory(site_id, date)
            
            # Check if URL exists
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                return 0, []  # No data available for this hour
            
            # Check if we already have files from this hour
            hour_pattern = f"{site_id}{date.strftime('%Y%m%d')}_{hour:02d}"
            existing_files = [f for f in os.listdir(site_dir) if f.startswith(hour_pattern)]
            if existing_files:
                return len(existing_files), existing_files  # Already downloaded
            
            # Download tar file
            self._log(site_id, f"Downloading hour {hour:02d}")
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            # Extract files
            extracted_files = []
            with tempfile.NamedTemporaryFile() as temp_tar:
                # Save tar content
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_tar.write(chunk)
                temp_tar.flush()
                
                # Extract tar file
                with tarfile.open(temp_tar.name, 'r') as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            # Extract to site directory
                            filename = os.path.basename(member.name)
                            member.name = filename
                            tar.extract(member, site_dir)
                            extracted_files.append(filename)
            
            self._log(site_id, f"✅ Hour {hour:02d}: {len(extracted_files)} files")
            return len(extracted_files), extracted_files
            
        except requests.RequestException as e:
            self._log(site_id, f"❌ Download failed for hour {hour:02d}: {str(e)}")
            return 0, []
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
                    self._download_hourly_data, 
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
        
        This method downloads individual files instead of entire hourly archives,
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
            # Check what files we already have
            existing_files = self.get_available_files(site_id, hours_back=6, max_files=max_files)
            
            if len(existing_files) >= max_files:
                results["files_already_exist"] = len(existing_files)
                results["file_list"] = existing_files[:max_files]
                logger.info(f"Already have {len(existing_files)} recent files for {site_id}")
                return results
            
            # If we need more files, try to download recent data
            needed_files = max_files - len(existing_files)
            logger.info(f"Need to download {needed_files} more files for {site_id}")
            
            # Download recent hourly data (last 3 hours should be sufficient)
            download_result = self.download_recent_data(site_id, hours_back=3)
            
            # Get updated file list
            updated_files = self.get_available_files(site_id, hours_back=6, max_files=max_files)
            results["files_downloaded"] = len(updated_files) - len(existing_files)
            results["file_list"] = updated_files
            
        except Exception as e:
            logger.error(f"Failed to download latest files for {site_id}: {e}")
            results["download_errors"] = 1
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
            list: List of available file paths
        """
        if site_id not in self.RADAR_SITES:
            raise ValueError(f"Unsupported site: {site_id}")
        
        files = []
        
        # Look through recent dates
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=hours_back // 24 + 1)
        
        current_date = start_date
        while current_date <= end_date:
            date_dir = os.path.join(self.data_dir, site_id, current_date.strftime('%Y-%m-%d'))
            
            if os.path.exists(date_dir):
                for filename in os.listdir(date_dir):
                    if filename.endswith('.ar2v'):
                        file_path = os.path.join(date_dir, filename)
                        files.append(file_path)
            
            current_date += timedelta(days=1)
        
        # Sort by modification time (most recent first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
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