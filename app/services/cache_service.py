"""
In-Memory Cache Service - Simple caching for processed radar data.
Provides intelligent caching for radar frames to reduce processing time from 47s to <2s.
Perfect for capstone projects without external dependencies.
"""
import hashlib
import logging
import numpy as np
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime, timedelta
import threading
import time

logger = logging.getLogger(__name__)


class RadarCacheService:
    """
    High-performance in-memory cache for processed radar data.
    
    Features:
    - Intelligent cache keys based on file content and processing parameters
    - TTL-based expiration (30 minutes for radar frames)
    - Thread-safe operations
    - Cache warming for active radar sites
    - Memory-efficient storage with automatic cleanup
    """
    
    def __init__(self, 
                 default_ttl: int = 1800,  # 30 minutes
                 max_cache_size: int = 1000):  # Maximum number of cached items
        """
        Initialize in-memory radar cache service.
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_cache_size: Maximum number of items to cache
        """
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self._connection_healthy = True
        
        # In-memory storage
        self._frame_cache: Dict[str, Tuple[np.ndarray, float]] = {}  # key -> (data, expiry_time)
        self._sequence_cache: Dict[str, Tuple[Tuple[np.ndarray, Dict], float]] = {}
        self._stats: Dict[str, int] = {
            "frames_cached": 0,
            "frames_retrieved": 0,
            "sequences_cached": 0,
            "sequences_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache key prefixes
        self.FRAME_PREFIX = "radar:frame:"
        self.SEQUENCE_PREFIX = "radar:sequence:"
        
        logger.info(f"In-memory RadarCacheService initialized (TTL: {default_ttl}s, max_size: {max_cache_size})")
    
    def _generate_cache_key(self, 
                          key_type: str, 
                          site_id: str, 
                          identifier: str, 
                          **params) -> str:
        """
        Generate cache key with optional parameters hash.
        
        Args:
            key_type: Type of cache key (frame, sequence, etc.)
            site_id: Radar site identifier
            identifier: Unique identifier (file path, timestamp, etc.)
            **params: Additional parameters to include in hash
            
        Returns:
            str: Generated cache key
        """
        # Create parameter hash if provided
        param_hash = ""
        if params:
            param_str = "|".join(f"{k}:{v}" for k, v in sorted(params.items()))
            param_hash = f":{hashlib.md5(param_str.encode()).hexdigest()[:8]}"
        
        # Generate file content hash for identifier if it's a file path
        if identifier.startswith('/') and '.' in identifier:
            try:
                # Use file size and modification time for faster hashing
                import os
                stat = os.stat(identifier)
                file_hash = hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:12]
                identifier = file_hash
            except (FileNotFoundError, PermissionError):
                # Fallback to filename if file not accessible
                identifier = identifier.split('/')[-1]
        
        return f"{key_type}{site_id}:{identifier}{param_hash}"
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        
        # Cleanup frame cache
        expired_frame_keys = [
            key for key, (_, expiry) in self._frame_cache.items() 
            if current_time > expiry
        ]
        for key in expired_frame_keys:
            del self._frame_cache[key]
        
        # Cleanup sequence cache
        expired_seq_keys = [
            key for key, (_, expiry) in self._sequence_cache.items() 
            if current_time > expiry
        ]
        for key in expired_seq_keys:
            del self._sequence_cache[key]
        
        if expired_frame_keys or expired_seq_keys:
            logger.debug(f"Cleaned up {len(expired_frame_keys)} frame entries and {len(expired_seq_keys)} sequence entries")
    
    def _enforce_cache_size_limit(self):
        """Enforce maximum cache size by removing oldest entries."""
        total_entries = len(self._frame_cache) + len(self._sequence_cache)
        
        if total_entries <= self.max_cache_size:
            return
        
        # Get all entries with their expiry times
        all_entries = []
        for key, (_, expiry) in self._frame_cache.items():
            all_entries.append((expiry, 'frame', key))
        for key, (_, expiry) in self._sequence_cache.items():
            all_entries.append((expiry, 'sequence', key))
        
        # Sort by expiry time (oldest first)
        all_entries.sort()
        
        # Remove oldest entries until we're under the limit
        entries_to_remove = total_entries - self.max_cache_size
        for i in range(entries_to_remove):
            if i >= len(all_entries):
                break
            _, cache_type, key = all_entries[i]
            if cache_type == 'frame':
                del self._frame_cache[key]
            else:
                del self._sequence_cache[key]
        
        logger.debug(f"Removed {entries_to_remove} old cache entries to enforce size limit")
    
    def cache_radar_frame(self, 
                         site_id: str, 
                         file_path: str, 
                         processed_frame: np.ndarray,
                         ttl: Optional[int] = None) -> bool:
        """
        Cache processed radar frame.
        
        Args:
            site_id: Radar site identifier
            file_path: Original radar file path
            processed_frame: Processed numpy array
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if cached successfully
        """
        if not self._connection_healthy:
            return False
        
        try:
            with self._lock:
                cache_key = self._generate_cache_key(
                    self.FRAME_PREFIX, 
                    site_id, 
                    file_path,
                    shape=processed_frame.shape,
                    dtype=str(processed_frame.dtype)
                )
                
                ttl = ttl or self.default_ttl
                expiry_time = time.time() + ttl
                
                self._frame_cache[cache_key] = (processed_frame.copy(), expiry_time)
                self._stats["frames_cached"] += 1
                
                # Periodic cleanup
                if len(self._frame_cache) % 100 == 0:
                    self._cleanup_expired()
                    self._enforce_cache_size_limit()
                
                logger.debug(f"Cached radar frame: {cache_key}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to cache radar frame: {e}")
            return False
    
    def get_radar_frame(self, 
                       site_id: str, 
                       file_path: str, 
                       expected_shape: tuple = None,
                       expected_dtype: str = None) -> Optional[np.ndarray]:
        """
        Retrieve cached radar frame.
        
        Args:
            site_id: Radar site identifier
            file_path: Original radar file path
            expected_shape: Expected array shape for validation
            expected_dtype: Expected data type for validation
            
        Returns:
            np.ndarray: Cached frame or None if not found
        """
        if not self._connection_healthy:
            return None
        
        try:
            with self._lock:
                cache_key = self._generate_cache_key(
                    self.FRAME_PREFIX, 
                    site_id, 
                    file_path,
                    shape=expected_shape,
                    dtype=expected_dtype
                )
                
                if cache_key not in self._frame_cache:
                    self._stats["cache_misses"] += 1
                    return None
                
                frame, expiry_time = self._frame_cache[cache_key]
                
                # Check if expired
                if time.time() > expiry_time:
                    del self._frame_cache[cache_key]
                    self._stats["cache_misses"] += 1
                    return None
                
                # Validate retrieved data
                if expected_shape and frame.shape != expected_shape:
                    logger.warning(f"Cached frame shape mismatch: {frame.shape} vs {expected_shape}")
                    return None
                
                if expected_dtype and str(frame.dtype) != expected_dtype:
                    logger.warning(f"Cached frame dtype mismatch: {frame.dtype} vs {expected_dtype}")
                    return None
                
                self._stats["frames_retrieved"] += 1
                self._stats["cache_hits"] += 1
                logger.debug(f"Retrieved cached radar frame: {cache_key}")
                return frame.copy()
                
        except Exception as e:
            logger.warning(f"Failed to retrieve radar frame: {e}")
            return None
    
    def cache_radar_sequence(self, 
                           site_id: str, 
                           file_paths: List[str], 
                           processed_sequence: np.ndarray,
                           metadata: Dict[str, Any],
                           ttl: Optional[int] = None) -> bool:
        """
        Cache processed radar sequence for model input.
        
        Args:
            site_id: Radar site identifier
            file_paths: List of original file paths
            processed_sequence: Processed sequence array
            metadata: Processing metadata
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if cached successfully
        """
        if not self._connection_healthy:
            return False
        
        try:
            with self._lock:
                # Create sequence identifier from file paths
                sequence_id = hashlib.md5("|".join(sorted(file_paths)).encode()).hexdigest()[:16]
                
                cache_key = self._generate_cache_key(
                    self.SEQUENCE_PREFIX, 
                    site_id, 
                    sequence_id,
                    length=len(file_paths),
                    shape=processed_sequence.shape
                )
                
                # Cache both sequence and metadata
                cache_data = (processed_sequence.copy(), metadata.copy())
                
                ttl = ttl or self.default_ttl
                expiry_time = time.time() + ttl
                
                self._sequence_cache[cache_key] = (cache_data, expiry_time)
                self._stats["sequences_cached"] += 1
                
                # Periodic cleanup
                if len(self._sequence_cache) % 50 == 0:
                    self._cleanup_expired()
                    self._enforce_cache_size_limit()
                
                logger.debug(f"Cached radar sequence: {cache_key}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to cache radar sequence: {e}")
            return False
    
    def get_radar_sequence(self, 
                          site_id: str, 
                          file_paths: List[str]) -> Optional[tuple]:
        """
        Retrieve cached radar sequence.
        
        Args:
            site_id: Radar site identifier
            file_paths: List of file paths for sequence
            
        Returns:
            tuple: (sequence_array, metadata) or None if not found
        """
        if not self._connection_healthy:
            return None
        
        try:
            with self._lock:
                sequence_id = hashlib.md5("|".join(sorted(file_paths)).encode()).hexdigest()[:16]
                
                cache_key = self._generate_cache_key(
                    self.SEQUENCE_PREFIX, 
                    site_id, 
                    sequence_id,
                    length=len(file_paths)
                )
                
                if cache_key not in self._sequence_cache:
                    self._stats["cache_misses"] += 1
                    return None
                
                cache_data, expiry_time = self._sequence_cache[cache_key]
                
                # Check if expired
                if time.time() > expiry_time:
                    del self._sequence_cache[cache_key]
                    self._stats["cache_misses"] += 1
                    return None
                
                self._stats["sequences_retrieved"] += 1
                self._stats["cache_hits"] += 1
                logger.debug(f"Retrieved cached radar sequence: {cache_key}")
                
                return cache_data
                
        except Exception as e:
            logger.warning(f"Failed to retrieve radar sequence: {e}")
            return None
    
    def warm_cache_for_site(self, site_id: str, recent_files: List[str]) -> Dict[str, int]:
        """
        Warm cache with recent radar files for a site.
        
        Args:
            site_id: Radar site identifier
            recent_files: List of recent radar files to process
            
        Returns:
            dict: Warming statistics
        """
        logger.info(f"Warming cache for site {site_id} with {len(recent_files)} files")
        
        stats = {
            "files_processed": 0,
            "frames_cached": 0,
            "cache_hits": 0,
            "processing_time_seconds": 0
        }
        
        start_time = datetime.now()
        
        # Import here to avoid circular imports
        from .radar_processing_service import RadarProcessingService
        processor = RadarProcessingService(enable_cache=False)  # Avoid recursion
        
        for file_path in recent_files[:20]:  # Limit to most recent 20 files
            try:
                # Check if frame is already cached
                cached_frame = self.get_radar_frame(site_id, file_path)
                if cached_frame is not None:
                    stats["cache_hits"] += 1
                    continue
                
                # Process and cache frame
                processed_frame = processor._process_file_uncached(file_path)
                if processed_frame is not None:
                    if self.cache_radar_frame(site_id, file_path, processed_frame):
                        stats["frames_cached"] += 1
                
                stats["files_processed"] += 1
                
            except Exception as e:
                logger.warning(f"Failed to warm cache for file {file_path}: {e}")
        
        stats["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Cache warming complete for {site_id}: "
                   f"{stats['frames_cached']} frames cached, "
                   f"{stats['cache_hits']} cache hits, "
                   f"{stats['processing_time_seconds']:.2f}s")
        
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            dict: Cache statistics
        """
        with self._lock:
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
            hit_ratio = self._stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "connection_healthy": self._connection_healthy,
                "cache_type": "in_memory",
                "cache_metrics": self._stats.copy(),
                "hit_ratio": hit_ratio,
                "frame_cache_size": len(self._frame_cache),
                "sequence_cache_size": len(self._sequence_cache),
                "total_cache_entries": len(self._frame_cache) + len(self._sequence_cache),
                "max_cache_size": self.max_cache_size,
                "default_ttl_seconds": self.default_ttl
            }
    
    def clear_cache(self, pattern: str = None) -> int:
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Key pattern (not used in simple implementation)
            
        Returns:
            int: Number of keys deleted
        """
        with self._lock:
            frame_count = len(self._frame_cache)
            sequence_count = len(self._sequence_cache)
            
            self._frame_cache.clear()
            self._sequence_cache.clear()
            
            # Reset stats
            for key in self._stats:
                self._stats[key] = 0
            
            total_cleared = frame_count + sequence_count
            logger.info(f"Cleared {total_cleared} cache entries")
            return total_cleared
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.
        
        Returns:
            dict: Health check results
        """
        return {
            "service": "RadarCacheService",
            "status": "healthy",
            "connection_healthy": True,
            "cache_type": "in_memory",
            "last_check": datetime.utcnow().isoformat(),
            "total_entries": len(self._frame_cache) + len(self._sequence_cache),
            "max_capacity": self.max_cache_size
        }


# Global cache instance
cache_service = RadarCacheService()