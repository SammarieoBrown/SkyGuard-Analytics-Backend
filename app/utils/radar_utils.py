"""
Radar Utilities - Helper functions for radar data manipulation and validation.
"""
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


def validate_radar_sequence(data: np.ndarray, expected_shape: Tuple[int, ...] = None) -> Dict[str, Any]:
    """
    Validate radar sequence data for model compatibility.
    
    Args:
        data: Radar sequence array
        expected_shape: Expected shape (optional)
        
    Returns:
        dict: Validation results
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "shape": list(data.shape),
        "data_type": str(data.dtype),
        "value_range": [float(data.min()), float(data.max())],
        "has_nan": bool(np.isnan(data).any()),
        "has_inf": bool(np.isinf(data).any())
    }
    
    # Check for expected shape
    if expected_shape and data.shape != expected_shape:
        validation["is_valid"] = False
        validation["issues"].append(f"Shape mismatch: got {data.shape}, expected {expected_shape}")
    
    # Check data type
    if data.dtype not in [np.float32, np.float64]:
        validation["issues"].append(f"Non-float data type: {data.dtype}")
    
    # Check value range (should be normalized [0,1])
    if data.min() < -0.01 or data.max() > 1.01:
        validation["issues"].append(f"Values outside [0,1] range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Check for invalid values
    if validation["has_nan"]:
        validation["is_valid"] = False
        validation["issues"].append("Data contains NaN values")
    
    if validation["has_inf"]:
        validation["is_valid"] = False
        validation["issues"].append("Data contains infinite values")
    
    return validation


def normalize_radar_data(data: np.ndarray, method: str = "min_max") -> np.ndarray:
    """
    Normalize radar data to [0, 1] range.
    
    Args:
        data: Input radar data
        method: Normalization method ("min_max" or "standard")
        
    Returns:
        np.ndarray: Normalized data
    """
    if method == "min_max":
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        else:
            return data
    elif method == "standard":
        return (data - data.mean()) / (data.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_timestamp_from_nexrad_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from NEXRAD filename.
    
    Args:
        filename: NEXRAD file name
        
    Returns:
        datetime: Extracted timestamp or None if parsing fails
    """
    try:
        # NEXRAD filename format: SITE_YYYYMMDD_HHMMSS_V03
        parts = filename.split('_')
        if len(parts) >= 3:
            date_str = parts[1]  # YYYYMMDD
            time_str = parts[2]  # HHMMSS
            
            timestamp_str = f"{date_str}_{time_str}"
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        pass
    
    return None


def sort_files_by_timestamp(file_paths: List[str]) -> List[str]:
    """
    Sort NEXRAD files by timestamp extracted from filename.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        list: Sorted file paths (chronological order)
    """
    def get_timestamp(filepath):
        filename = os.path.basename(filepath)
        timestamp = extract_timestamp_from_nexrad_filename(filename)
        return timestamp or datetime.min
    
    return sorted(file_paths, key=get_timestamp)


def create_prediction_timestamps(base_time: datetime, num_frames: int, interval_minutes: int = 10) -> List[str]:
    """
    Create timestamp labels for prediction frames.
    
    Args:
        base_time: Starting time
        num_frames: Number of prediction frames
        interval_minutes: Minutes between frames
        
    Returns:
        list: Formatted timestamp strings
    """
    from datetime import timedelta
    
    timestamps = []
    for i in range(num_frames):
        frame_time = base_time + timedelta(minutes=interval_minutes * (i + 1))
        timestamps.append(frame_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    return timestamps


def calculate_data_freshness(file_path: str) -> Dict[str, Any]:
    """
    Calculate how fresh radar data is based on file modification time.
    
    Args:
        file_path: Path to the radar file
        
    Returns:
        dict: Freshness information
    """
    try:
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        current_time = datetime.now()
        age_seconds = (current_time - file_time).total_seconds()
        age_hours = age_seconds / 3600
        
        # Determine freshness category
        if age_hours < 1:
            category = "excellent"
        elif age_hours < 6:
            category = "good"
        elif age_hours < 24:
            category = "fair"
        else:
            category = "poor"
        
        return {
            "file_time": file_time,
            "current_time": current_time,
            "age_seconds": age_seconds,
            "age_hours": age_hours,
            "category": category
        }
        
    except OSError:
        return {
            "error": "File not accessible",
            "category": "unknown"
        }


def convert_polar_to_cartesian(azimuth: np.ndarray, 
                             range_vals: np.ndarray, 
                             data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert polar radar coordinates to Cartesian coordinates.
    
    Args:
        azimuth: Azimuth angles in degrees
        range_vals: Range values in km
        data: Radar data values
        
    Returns:
        tuple: (x_coords, y_coords, cartesian_data)
    """
    # Convert to radians
    az_rad = np.deg2rad(azimuth)
    
    # Create coordinate grids
    range_grid, az_grid = np.meshgrid(range_vals, az_rad)
    
    # Convert to Cartesian
    x_coords = range_grid * np.sin(az_grid)
    y_coords = range_grid * np.cos(az_grid)
    
    return x_coords, y_coords, data


def get_radar_coverage_info(site_id: str) -> Dict[str, Any]:
    """
    Get radar coverage information for a site.
    
    Args:
        site_id: Radar site identifier
        
    Returns:
        dict: Coverage information
    """
    # Standard NEXRAD coverage parameters
    coverage_info = {
        "max_range_km": 460,  # Maximum range
        "effective_range_km": 230,  # Effective precipitation range
        "resolution_meters": 250,  # Range resolution
        "elevation_angles": list(range(1, 15)),  # Typical elevation angles
        "scan_time_minutes": 5,  # Typical volume scan time
        "beam_width_degrees": 1.0  # Beam width
    }
    
    # Site-specific adjustments
    site_adjustments = {
        "KAMX": {
            "terrain": "coastal",
            "elevation_meters": 8,
            "notes": "May have ground clutter from buildings"
        },
        "KATX": {
            "terrain": "mountainous",
            "elevation_meters": 151,
            "notes": "Mountain blocking may affect coverage at low elevations"
        }
    }
    
    if site_id in site_adjustments:
        coverage_info.update(site_adjustments[site_id])
    
    return coverage_info


def estimate_precipitation_intensity(reflectivity_dbz: np.ndarray) -> np.ndarray:
    """
    Estimate precipitation intensity from reflectivity values.
    
    Args:
        reflectivity_dbz: Reflectivity in dBZ
        
    Returns:
        np.ndarray: Precipitation rate in mm/hr
    """
    # Marshall-Palmer Z-R relationship: Z = 200 * R^1.6
    # Solving for R: R = (Z / 200)^(1/1.6)
    
    # Convert dBZ to linear Z
    z_linear = 10 ** (reflectivity_dbz / 10)
    
    # Apply Z-R relationship
    precip_rate = np.power(z_linear / 200.0, 1.0 / 1.6)
    
    # Handle invalid values
    precip_rate[reflectivity_dbz < 5] = 0  # Below detection threshold
    precip_rate[np.isnan(reflectivity_dbz)] = 0
    
    return precip_rate


def create_model_summary(model_input_shape: Tuple[int, ...], 
                        model_output_shape: Tuple[int, ...],
                        processing_time_ms: float) -> Dict[str, Any]:
    """
    Create a summary of model processing information.
    
    Args:
        model_input_shape: Shape of model input
        model_output_shape: Shape of model output
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        dict: Model summary information
    """
    return {
        "input_shape": list(model_input_shape),
        "output_shape": list(model_output_shape),
        "processing_time_ms": processing_time_ms,
        "input_sequence_length": model_input_shape[1] if len(model_input_shape) > 1 else None,
        "output_sequence_length": model_output_shape[1] if len(model_output_shape) > 1 else None,
        "spatial_resolution": f"{model_input_shape[2]}x{model_input_shape[3]}" if len(model_input_shape) > 3 else None,
        "performance_category": "fast" if processing_time_ms < 1000 else "moderate" if processing_time_ms < 5000 else "slow"
    }


def validate_file_path(file_path: str, required_extension: str = ".ar2v") -> bool:
    """
    Validate that a file path exists and has the correct extension.
    
    Args:
        file_path: Path to validate
        required_extension: Required file extension
        
    Returns:
        bool: True if valid, False otherwise
    """
    path = Path(file_path)
    return (path.exists() and 
            path.is_file() and 
            path.suffix.lower() == required_extension.lower() and
            path.stat().st_size > 0)


def get_storage_usage(directory: str) -> Dict[str, Any]:
    """
    Get storage usage information for a directory.
    
    Args:
        directory: Directory path to analyze
        
    Returns:
        dict: Storage usage information
    """
    try:
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    file_count += 1
                except OSError:
                    continue
        
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "file_count": file_count,
            "directory": directory,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "directory": directory,
            "status": "error"
        }