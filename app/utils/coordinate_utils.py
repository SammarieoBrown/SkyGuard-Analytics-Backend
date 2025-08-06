"""
Coordinate Utilities for Radar Data Processing.

Provides geographic coordinate transformations and metadata calculation
for radar data visualization and API responses.
"""
import numpy as np
from typing import Tuple, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RadarCoordinateCalculator:
    """
    Utility class for calculating geographic coordinates from radar data.
    
    Handles coordinate transformations, bounds calculation, and metadata
    generation for radar visualization and API responses.
    """
    
    def __init__(self, range_km: int = 150):
        """
        Initialize coordinate calculator.
        
        Args:
            range_km: Radar range in kilometers (default: 150km)
        """
        self.range_km = range_km
        self.earth_radius_km = 6371.0  # Earth's radius in km
        
    def calculate_radar_bounds(self, 
                              site_lat: float, 
                              site_lon: float, 
                              data_shape: Tuple[int, int] = (64, 64)) -> Dict[str, Any]:
        """
        Calculate geographic bounds and coordinate metadata for radar data.
        
        Args:
            site_lat: Radar site latitude
            site_lon: Radar site longitude  
            data_shape: Shape of the radar data array (height, width)
            
        Returns:
            dict: Coordinate metadata including bounds, resolution, etc.
        """
        try:
            # Calculate approximate degrees per kilometer at this latitude
            lat_deg_per_km = 1.0 / 111.0
            lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians(site_lat)))
            
            # Calculate bounds
            lat_range = self.range_km * lat_deg_per_km
            lon_range = self.range_km * lon_deg_per_km
            
            bounds = [
                site_lon - lon_range,  # west
                site_lon + lon_range,  # east
                site_lat - lat_range,  # south
                site_lat + lat_range   # north
            ]
            
            # Calculate resolution (degrees per pixel)
            resolution_deg_lat = (2 * lat_range) / data_shape[0]
            resolution_deg_lon = (2 * lon_range) / data_shape[1]
            resolution_deg = max(resolution_deg_lat, resolution_deg_lon)  # Use larger for square pixels
            
            # Calculate km per pixel
            resolution_km = (2 * self.range_km) / max(data_shape)
            
            return {
                "bounds": bounds,
                "center": [site_lat, site_lon],
                "resolution_deg": float(resolution_deg),
                "resolution_km": float(resolution_km),
                "projection": "PlateCarree",
                "range_km": self.range_km,
                "data_shape": list(data_shape)
            }
            
        except Exception as e:
            logger.error(f"Error calculating radar bounds: {e}")
            raise ValueError(f"Failed to calculate coordinate bounds: {str(e)}")
    
    def pixels_to_coordinates(self, 
                             pixel_coords: List[Tuple[int, int]],
                             site_lat: float,
                             site_lon: float,
                             data_shape: Tuple[int, int] = (64, 64)) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel_coords: List of (row, col) pixel coordinates
            site_lat: Radar site latitude
            site_lon: Radar site longitude
            data_shape: Shape of the radar data array
            
        Returns:
            list: List of (lat, lon) geographic coordinates
        """
        try:
            bounds_info = self.calculate_radar_bounds(site_lat, site_lon, data_shape)
            bounds = bounds_info["bounds"]
            
            # Calculate coordinate ranges
            lon_range = bounds[1] - bounds[0]  # east - west
            lat_range = bounds[3] - bounds[2]  # north - south
            
            geo_coords = []
            for row, col in pixel_coords:
                # Convert pixel to geographic coordinates
                lon = bounds[0] + (col / data_shape[1]) * lon_range
                lat = bounds[3] - (row / data_shape[0]) * lat_range  # Flip Y axis
                geo_coords.append((lat, lon))
            
            return geo_coords
            
        except Exception as e:
            logger.error(f"Error converting pixels to coordinates: {e}")
            raise ValueError(f"Failed to convert pixel coordinates: {str(e)}")
    
    def coordinates_to_pixels(self,
                             geo_coords: List[Tuple[float, float]],
                             site_lat: float,
                             site_lon: float,
                             data_shape: Tuple[int, int] = (64, 64)) -> List[Tuple[int, int]]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            geo_coords: List of (lat, lon) geographic coordinates
            site_lat: Radar site latitude
            site_lon: Radar site longitude
            data_shape: Shape of the radar data array
            
        Returns:
            list: List of (row, col) pixel coordinates
        """
        try:
            bounds_info = self.calculate_radar_bounds(site_lat, site_lon, data_shape)
            bounds = bounds_info["bounds"]
            
            # Calculate coordinate ranges
            lon_range = bounds[1] - bounds[0]  # east - west
            lat_range = bounds[3] - bounds[2]  # north - south
            
            pixel_coords = []
            for lat, lon in geo_coords:
                # Convert geographic to pixel coordinates
                col = int(((lon - bounds[0]) / lon_range) * data_shape[1])
                row = int(((bounds[3] - lat) / lat_range) * data_shape[0])  # Flip Y axis
                
                # Clamp to valid pixel range
                col = max(0, min(col, data_shape[1] - 1))
                row = max(0, min(row, data_shape[0] - 1))
                
                pixel_coords.append((row, col))
            
            return pixel_coords
            
        except Exception as e:
            logger.error(f"Error converting coordinates to pixels: {e}")
            raise ValueError(f"Failed to convert geographic coordinates: {str(e)}")
    
    def calculate_composite_bounds(self, site_locations: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Calculate composite bounds that encompass multiple radar sites.
        
        Args:
            site_locations: Dictionary of site_id -> (lat, lon) pairs
            
        Returns:
            dict: Composite coordinate metadata
        """
        try:
            if not site_locations:
                raise ValueError("No site locations provided")
            
            # Calculate individual site bounds
            all_bounds = []
            for site_id, (lat, lon) in site_locations.items():
                site_bounds = self.calculate_radar_bounds(lat, lon)
                all_bounds.append(site_bounds["bounds"])
            
            # Find overall bounds (min west, max east, min south, max north)
            composite_bounds = [
                min(bounds[0] for bounds in all_bounds),  # westernmost
                max(bounds[1] for bounds in all_bounds),  # easternmost  
                min(bounds[2] for bounds in all_bounds),  # southernmost
                max(bounds[3] for bounds in all_bounds)   # northernmost
            ]
            
            # Calculate composite center
            center_lat = (composite_bounds[2] + composite_bounds[3]) / 2
            center_lon = (composite_bounds[0] + composite_bounds[1]) / 2
            
            # Calculate composite resolution
            lat_range = composite_bounds[3] - composite_bounds[2]
            lon_range = composite_bounds[1] - composite_bounds[0]
            max_range_deg = max(lat_range, lon_range)
            
            return {
                "bounds": composite_bounds,
                "center": [center_lat, center_lon],
                "resolution_deg": max_range_deg / 400,  # Assume 400px composite
                "resolution_km": (max_range_deg * 111.0) / 400,
                "projection": "PlateCarree",
                "range_km": int(max_range_deg * 111.0 / 2),
                "coverage_sites": list(site_locations.keys())
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite bounds: {e}")
            raise ValueError(f"Failed to calculate composite bounds: {str(e)}")
    
    def validate_coordinates(self, coordinates: Dict[str, Any]) -> bool:
        """
        Validate coordinate metadata structure and values.
        
        Args:
            coordinates: Coordinate metadata dictionary
            
        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        required_fields = ["bounds", "center", "resolution_deg", "resolution_km"]
        
        for field in required_fields:
            if field not in coordinates:
                raise ValueError(f"Missing required coordinate field: {field}")
        
        bounds = coordinates["bounds"]
        if len(bounds) != 4:
            raise ValueError("Bounds must have 4 values [west, east, south, north]")
        
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("Invalid bounds: west >= east or south >= north")
        
        center = coordinates["center"]
        if len(center) != 2:
            raise ValueError("Center must have 2 values [lat, lon]")
        
        if not (-90 <= center[0] <= 90):
            raise ValueError("Center latitude must be between -90 and 90")
        
        if not (-180 <= center[1] <= 180):
            raise ValueError("Center longitude must be between -180 and 180")
        
        if coordinates["resolution_deg"] <= 0 or coordinates["resolution_km"] <= 0:
            raise ValueError("Resolution values must be positive")
        
        return True
    
    def get_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            float: Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return self.earth_radius_km * c


# Convenience functions for backward compatibility and ease of use

def calculate_radar_coordinates(site_lat: float, 
                               site_lon: float, 
                               data_shape: Tuple[int, int] = (64, 64),
                               range_km: int = 150) -> Dict[str, Any]:
    """
    Calculate coordinate metadata for a radar site (convenience function).
    
    Args:
        site_lat: Radar site latitude
        site_lon: Radar site longitude
        data_shape: Shape of radar data array (height, width)
        range_km: Radar range in kilometers
        
    Returns:
        dict: Coordinate metadata
    """
    calculator = RadarCoordinateCalculator(range_km)
    return calculator.calculate_radar_bounds(site_lat, site_lon, data_shape)


def get_site_coordinates(site_id: str) -> Tuple[float, float]:
    """
    Get latitude and longitude for a supported radar site.
    
    Args:
        site_id: Radar site identifier (KAMX, KATX)
        
    Returns:
        tuple: (latitude, longitude)
    """
    # Radar site coordinates (matches NEXRADDataService.RADAR_SITES)
    site_coords = {
        "KAMX": (25.6112, -80.4128),  # Miami, FL
        "KATX": (48.1947, -122.4956)  # Seattle, WA
    }
    
    if site_id not in site_coords:
        raise ValueError(f"Unsupported site: {site_id}. Supported: {list(site_coords.keys())}")
    
    return site_coords[site_id]


def create_coordinate_metadata(site_id: str, 
                              data_shape: Tuple[int, int] = (64, 64),
                              range_km: int = 150) -> Dict[str, Any]:
    """
    Create complete coordinate metadata for a radar site (convenience function).
    
    Args:
        site_id: Radar site identifier
        data_shape: Shape of radar data array
        range_km: Radar range in kilometers
        
    Returns:
        dict: Complete coordinate metadata
    """
    lat, lon = get_site_coordinates(site_id)
    return calculate_radar_coordinates(lat, lon, data_shape, range_km)