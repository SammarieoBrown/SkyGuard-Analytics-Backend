"""
Radar Mosaic Service - Create NWS-style weather maps from NEXRAD data.
Generates professional weather visualizations similar to the NWS Radar Mosaic.
"""
import os
import io
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import pyart

logger = logging.getLogger(__name__)


class RadarMosaicService:
    """
    Service for creating weather radar mosaics and visualizations.
    
    Features:
    - NWS-style radar composites
    - Geographic projections with Cartopy
    - Custom colormaps matching NWS standards
    - Multi-site radar compositing
    - Regional and CONUS views
    """
    
    def __init__(self):
        """Initialize radar mosaic service."""
        self.supported_sites = {
            "KAMX": {"name": "Miami", "lat": 25.6112, "lon": -80.4128},
            "KATX": {"name": "Seattle", "lat": 48.1947, "lon": -122.4956}
        }
        
        # NWS-style reflectivity colormap
        self.nws_colors = [
            '#FFFFFF',  # No data/clear
            '#9C9C9C',  # Light gray
            '#767676',  # Medium gray
            '#00ECEC',  # Light blue
            '#01A0F6',  # Blue
            '#0000F6',  # Dark blue
            '#00FF00',  # Green
            '#00BB00',  # Dark green
            '#FFFF00',  # Yellow
            '#FFD800',  # Gold
            '#FF9500',  # Orange
            '#FF0000',  # Red
            '#D60000',  # Dark red
            '#C00000',  # Maroon
            '#FF00FF',  # Magenta
            '#9955C4'   # Purple
        ]
        
        # Reflectivity levels (dBZ)
        self.dbz_levels = [-32, -20, -10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        
        logger.info("RadarMosaicService initialized")
    
    def create_nws_colormap(self) -> mcolors.ListedColormap:
        """
        Create NWS-style colormap for reflectivity data.
        
        Returns:
            matplotlib.colors.ListedColormap: NWS-style colormap
        """
        return mcolors.ListedColormap(self.nws_colors)
    
    def create_site_visualization(self, 
                                 site_id: str, 
                                 processed_frame: np.ndarray,
                                 extent: Tuple[float, float, float, float] = None,
                                 show_range_rings: bool = True,
                                 title: str = None) -> bytes:
        """
        Create a single-site radar visualization.
        
        Args:
            site_id: Radar site identifier
            processed_frame: Processed radar data (64x64 array)
            extent: Map extent (lon_min, lon_max, lat_min, lat_max)
            show_range_rings: Show range rings around radar
            title: Custom title for the plot
            
        Returns:
            bytes: PNG image data
        """
        if site_id not in self.supported_sites:
            raise ValueError(f"Unsupported site: {site_id}")
        
        site_info = self.supported_sites[site_id]
        
        # Set default extent around the radar site
        if extent is None:
            range_km = 150  # 150km range
            lat_range = range_km / 111.0  # Approximate degrees
            lon_range = range_km / (111.0 * np.cos(np.radians(site_info["lat"])))
            extent = (
                site_info["lon"] - lon_range,
                site_info["lon"] + lon_range,
                site_info["lat"] - lat_range,
                site_info["lat"] + lat_range
            )
        
        # Create figure with map projection
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')
        ax.add_feature(cfeature.RIVERS, alpha=0.3, edgecolor='lightblue')
        
        # Convert processed frame to reflectivity-like values
        # Assuming processed_frame is normalized 0-255, convert to dBZ range
        dbz_data = (processed_frame.astype(np.float32) / 255.0) * 127 - 32  # Scale to -32 to +95 dBZ
        
        # Create coordinate arrays for the radar data
        range_km = 150
        x_coords = np.linspace(-range_km, range_km, processed_frame.shape[1])
        y_coords = np.linspace(-range_km, range_km, processed_frame.shape[0])
        
        # Convert km to lat/lon
        lat_coords = site_info["lat"] + y_coords / 111.0
        lon_coords = site_info["lon"] + x_coords / (111.0 * np.cos(np.radians(site_info["lat"])))
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Plot radar data
        nws_cmap = self.create_nws_colormap()
        radar_plot = ax.pcolormesh(
            lon_grid, lat_grid, dbz_data,
            cmap=nws_cmap,
            vmin=self.dbz_levels[0],
            vmax=self.dbz_levels[-1],
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )
        
        # Add radar site marker
        ax.plot(site_info["lon"], site_info["lat"], 'ro', markersize=8, 
               transform=ccrs.PlateCarree(), label=f"{site_id} Radar")
        
        # Add range rings if requested
        if show_range_rings:
            for range_km in [50, 100, 150]:
                lat_range = range_km / 111.0
                lon_range = range_km / (111.0 * np.cos(np.radians(site_info["lat"])))
                
                circle = Circle(
                    (site_info["lon"], site_info["lat"]),
                    max(lat_range, lon_range),
                    fill=False, color='white', linewidth=1, alpha=0.6,
                    transform=ccrs.PlateCarree()
                )
                ax.add_patch(circle)
        
        # Add colorbar
        cbar = plt.colorbar(radar_plot, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Reflectivity (dBZ)', rotation=270, labelpad=20)
        cbar.set_ticks(self.dbz_levels[::2])  # Every other level
        
        # Add title
        if title is None:
            title = f"NEXRAD {site_id} - {site_info['name']} Radar"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        ax.set_title(f"{title}\n{current_time}", fontsize=14, pad=20)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        # Convert to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        plt.close(fig)
        
        return image_data
    
    def create_composite_mosaic(self, 
                               site_data: Dict[str, np.ndarray],
                               extent: Tuple[float, float, float, float] = None,
                               title: str = "NEXRAD Composite Mosaic") -> bytes:
        """
        Create a composite radar mosaic from multiple sites.
        
        Args:
            site_data: Dictionary of site_id -> processed_frame
            extent: Map extent (lon_min, lon_max, lat_min, lat_max)
            title: Title for the mosaic
            
        Returns:
            bytes: PNG image data
        """
        if not site_data:
            raise ValueError("No site data provided for composite")
        
        # Set default CONUS extent if not provided
        if extent is None:
            extent = (-130, -65, 20, 50)  # CONUS extent
        
        # Create figure with map projection
        fig = plt.figure(figsize=(16, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black', alpha=0.7)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue')
        ax.add_feature(cfeature.RIVERS, alpha=0.3, edgecolor='lightblue')
        
        # Create composite grid
        grid_size = (400, 600)  # Higher resolution for composite
        composite = np.zeros(grid_size, dtype=np.float32)
        composite_counts = np.zeros(grid_size, dtype=np.int32)
        
        # Grid coordinates
        lon_range = extent[1] - extent[0]
        lat_range = extent[3] - extent[2]
        
        nws_cmap = self.create_nws_colormap()
        
        # Process each site
        for site_id, processed_frame in site_data.items():
            if site_id not in self.supported_sites:
                logger.warning(f"Skipping unknown site: {site_id}")
                continue
            
            site_info = self.supported_sites[site_id]
            
            # Convert processed frame to reflectivity
            dbz_data = (processed_frame.astype(np.float32) / 255.0) * 127 - 32
            
            # Project site data onto composite grid
            range_km = 150
            for i in range(processed_frame.shape[0]):
                for j in range(processed_frame.shape[1]):
                    # Calculate real-world coordinates
                    y_km = (i - processed_frame.shape[0] // 2) * (2 * range_km / processed_frame.shape[0])
                    x_km = (j - processed_frame.shape[1] // 2) * (2 * range_km / processed_frame.shape[1])
                    
                    # Convert to lat/lon
                    lat = site_info["lat"] + y_km / 111.0
                    lon = site_info["lon"] + x_km / (111.0 * np.cos(np.radians(site_info["lat"])))
                    
                    # Check if within composite extent
                    if extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]:
                        # Convert to grid coordinates
                        grid_x = int((lon - extent[0]) / lon_range * grid_size[1])
                        grid_y = int((extent[3] - lat) / lat_range * grid_size[0])
                        
                        if 0 <= grid_x < grid_size[1] and 0 <= grid_y < grid_size[0]:
                            value = dbz_data[i, j]
                            if value > -30:  # Only include significant reflectivity
                                composite[grid_y, grid_x] += value
                                composite_counts[grid_y, grid_x] += 1
            
            # Add site marker
            ax.plot(site_info["lon"], site_info["lat"], 'ro', markersize=6, 
                   transform=ccrs.PlateCarree(), 
                   label=f"{site_id}")
        
        # Average overlapping values
        valid_mask = composite_counts > 0
        composite[valid_mask] /= composite_counts[valid_mask]
        composite[~valid_mask] = np.nan
        
        # Create coordinate arrays for plotting
        lon_coords = np.linspace(extent[0], extent[1], grid_size[1])
        lat_coords = np.linspace(extent[3], extent[2], grid_size[0])
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Plot composite data
        composite_plot = ax.pcolormesh(
            lon_grid, lat_grid, composite,
            cmap=nws_cmap,
            vmin=self.dbz_levels[0],
            vmax=self.dbz_levels[-1],
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )
        
        # Add colorbar
        cbar = plt.colorbar(composite_plot, ax=ax, shrink=0.6, pad=0.05)
        cbar.set_label('Reflectivity (dBZ)', rotation=270, labelpad=20)
        cbar.set_ticks(self.dbz_levels[::2])
        
        # Add title and timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        ax.set_title(f"{title}\n{current_time}", fontsize=16, pad=20)
        
        # Add legend
        ax.legend(loc='upper right', title="Radar Sites")
        
        # Add grid
        ax.gridlines(draw_labels=True, alpha=0.3)
        
        # Convert to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        plt.close(fig)
        
        return image_data
    
    def create_animation_frames(self, 
                               site_id: str, 
                               frame_sequence: List[np.ndarray],
                               output_dir: str = None) -> List[str]:
        """
        Create animation frames from a sequence of radar data.
        
        Args:
            site_id: Radar site identifier
            frame_sequence: List of processed radar frames
            output_dir: Directory to save frames (default: temp directory)
            
        Returns:
            list: List of frame file paths
        """
        if output_dir is None:
            import tempfile
            output_dir = tempfile.mkdtemp()
        
        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []
        
        for i, frame in enumerate(frame_sequence):
            title = f"NEXRAD {site_id} - Frame {i+1}/{len(frame_sequence)}"
            image_data = self.create_site_visualization(site_id, frame, title=title)
            
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
            with open(frame_path, 'wb') as f:
                f.write(image_data)
            
            frame_paths.append(frame_path)
        
        logger.info(f"Created {len(frame_paths)} animation frames in {output_dir}")
        return frame_paths
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get radar mosaic service information.
        
        Returns:
            dict: Service configuration and capabilities
        """
        return {
            "service": "RadarMosaicService",
            "supported_sites": list(self.supported_sites.keys()),
            "site_details": self.supported_sites,
            "capabilities": [
                "single_site_visualization",
                "composite_mosaic",
                "animation_frames",
                "nws_style_colormaps"
            ],
            "output_formats": ["PNG"],
            "projections": ["PlateCarree"],
            "colormap": "NWS_Reflectivity"
        }