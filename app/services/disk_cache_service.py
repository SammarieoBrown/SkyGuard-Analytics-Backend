"""
Disk Cache Service for processed radar frames.
Uses persistent disk on Render for fast local caching.
"""
import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from app.config import CACHE_DIR, IS_RENDER

logger = logging.getLogger(__name__)


class DiskCacheService:
    """
    Service for caching processed radar frames on disk.
    Provides fast local cache for frequently accessed data.
    """
    
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initialize disk cache service.
        
        Args:
            cache_ttl_hours: Time-to-live for cached items in hours
        """
        self.cache_dir = CACHE_DIR
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.enabled = False
        
        # Only enable on Render with persistent disk
        if IS_RENDER and self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.enabled = True
                logger.info(f"Disk cache enabled at {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize disk cache: {e}")
                self.enabled = False
        else:
            logger.info("Disk cache disabled (not on Render or no cache dir)")
    
    def _get_cache_key(self, site_id: str, filepath: str) -> str:
        """Generate cache key from site and file."""
        key_string = f"{site_id}:{filepath}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, site_id: str, cache_key: str) -> Path:
        """Get full path for cached file."""
        site_dir = self.cache_dir / "radar" / site_id
        site_dir.mkdir(parents=True, exist_ok=True)
        return site_dir / f"{cache_key}.npy"
    
    def _get_metadata_path(self, site_id: str, cache_key: str) -> Path:
        """Get path for cache metadata."""
        site_dir = self.cache_dir / "radar" / site_id
        return site_dir / f"{cache_key}.meta"
    
    def get_cached_frame(self, site_id: str, filepath: str) -> Optional[np.ndarray]:
        """
        Get cached processed frame if available and not expired.
        
        Args:
            site_id: Radar site ID
            filepath: Original file path or blob name
            
        Returns:
            Cached numpy array or None if not found/expired
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._get_cache_key(site_id, filepath)
            cache_path = self._get_cache_path(site_id, cache_key)
            meta_path = self._get_metadata_path(site_id, cache_key)
            
            # Check if cache files exist
            if not cache_path.exists() or not meta_path.exists():
                return None
            
            # Check metadata for expiry
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            cached_time = datetime.fromisoformat(metadata['cached_at'])
            if datetime.now() - cached_time > self.cache_ttl:
                # Cache expired, remove files
                cache_path.unlink()
                meta_path.unlink()
                return None
            
            # Load and return cached array
            array = np.load(cache_path)
            logger.debug(f"Disk cache hit for {site_id}/{os.path.basename(filepath)}")
            return array
            
        except Exception as e:
            logger.debug(f"Failed to get cached frame: {e}")
            return None
    
    def cache_frame(self, site_id: str, filepath: str, array: np.ndarray) -> bool:
        """
        Cache a processed frame to disk.
        
        Args:
            site_id: Radar site ID
            filepath: Original file path or blob name
            array: Processed numpy array
            
        Returns:
            True if cached successfully
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._get_cache_key(site_id, filepath)
            cache_path = self._get_cache_path(site_id, cache_key)
            meta_path = self._get_metadata_path(site_id, cache_key)
            
            # Save array
            np.save(cache_path, array)
            
            # Save metadata
            metadata = {
                'site_id': site_id,
                'filepath': filepath,
                'cached_at': datetime.now().isoformat(),
                'shape': array.shape,
                'dtype': str(array.dtype)
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Cached frame to disk: {site_id}/{os.path.basename(filepath)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache frame: {e}")
            return False
    
    def clear_expired_cache(self) -> Dict[str, int]:
        """
        Remove expired cache entries.
        
        Returns:
            Statistics about cleared cache
        """
        if not self.enabled:
            return {'cleared': 0, 'errors': 0}
        
        stats = {'cleared': 0, 'errors': 0}
        
        try:
            radar_dir = self.cache_dir / "radar"
            if not radar_dir.exists():
                return stats
            
            # Check all site directories
            for site_dir in radar_dir.iterdir():
                if not site_dir.is_dir():
                    continue
                
                # Check all metadata files
                for meta_file in site_dir.glob("*.meta"):
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                        
                        cached_time = datetime.fromisoformat(metadata['cached_at'])
                        if datetime.now() - cached_time > self.cache_ttl:
                            # Remove cache and metadata files
                            cache_file = meta_file.with_suffix('.npy')
                            if cache_file.exists():
                                cache_file.unlink()
                            meta_file.unlink()
                            stats['cleared'] += 1
                            
                    except Exception as e:
                        logger.debug(f"Error processing {meta_file}: {e}")
                        stats['errors'] += 1
            
            logger.info(f"Cleared {stats['cleared']} expired cache entries")
            
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            stats['errors'] += 1
        
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache usage statistics
        """
        if not self.enabled:
            return {'enabled': False}
        
        stats = {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_files': 0,
            'total_size_mb': 0,
            'sites': {}
        }
        
        try:
            radar_dir = self.cache_dir / "radar"
            if radar_dir.exists():
                for site_dir in radar_dir.iterdir():
                    if not site_dir.is_dir():
                        continue
                    
                    site_stats = {
                        'files': 0,
                        'size_mb': 0
                    }
                    
                    for cache_file in site_dir.glob("*.npy"):
                        site_stats['files'] += 1
                        site_stats['size_mb'] += cache_file.stat().st_size / (1024 * 1024)
                        stats['total_files'] += 1
                        stats['total_size_mb'] += cache_file.stat().st_size / (1024 * 1024)
                    
                    if site_stats['files'] > 0:
                        stats['sites'][site_dir.name] = site_stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            stats['error'] = str(e)
        
        return stats


# Global singleton instance
_disk_cache_instance: Optional[DiskCacheService] = None


def get_disk_cache() -> DiskCacheService:
    """Get or create the singleton disk cache service."""
    global _disk_cache_instance
    
    if _disk_cache_instance is None:
        _disk_cache_instance = DiskCacheService()
    
    return _disk_cache_instance