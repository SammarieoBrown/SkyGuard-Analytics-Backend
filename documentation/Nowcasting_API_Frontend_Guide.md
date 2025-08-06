# Nowcasting API Frontend Integration Guide

This guide provides comprehensive documentation for integrating with the SkyGuard Analytics Nowcasting API endpoints. All endpoints are designed for real-time weather radar data processing and visualization.

## Base URL
```
http://localhost:8000/api/v1/nowcasting
```

## Supported Radar Sites
- **KAMX** - Miami, FL (25.6112, -80.4128)
- **KATX** - Seattle, WA (48.1947, -122.4956)

---

## 🎯 **Core Weather Prediction Endpoints**

### 1. **POST** `/predict` - Predict Weather Nowcast

Generates 6-frame precipitation predictions based on latest radar data.

**Request Schema:**
```typescript
interface NowcastingPredictionRequest {
  site_id: "KAMX" | "KATX";                    // Required: Radar site
  use_latest_data?: boolean;                   // Default: true
  hours_back?: number;                         // 1-48, Default: 12
  custom_radar_data?: number[][][][];          // Optional: (10,64,64,1) array
}
```

**Example Request:**
```javascript
const response = await fetch('/api/v1/nowcasting/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    site_id: "KAMX",
    use_latest_data: true,
    hours_back: 6
  })
});
```

**Response Schema:**
```typescript
interface NowcastingPredictionResponse {
  success: boolean;
  site_info: {
    site_id: string;
    name: string;
    location: string;
    coordinates: [number, number];             // [lat, lon]
    description: string;
  };
  prediction_frames: number[][][][];          // 6 future frames (6,64,64,1)
  input_metadata: {
    data_source: "nexrad_gcp" | "custom";
    files_used?: number;
    processing_metadata?: object;
  };
  ml_model_metadata: object;
  processing_time_ms: number;
  prediction_timestamp: string;               // ISO datetime
  frame_times: string[];                      // 6 future timestamps
  confidence_metrics?: Record<string, number>;
}
```

**Usage:**
- Perfect for **weather forecasting dashboards**
- Returns **6 future radar frames** (10-minute intervals)
- Use `prediction_frames` for animated weather predictions

---

### 2. **POST** `/batch` - Batch Weather Nowcast

Generate predictions for multiple radar sites simultaneously.

**Request Schema:**
```typescript
interface BatchNowcastingRequest {
  site_ids: ("KAMX" | "KATX")[];             // Array of site IDs
  hours_back?: number;                       // 1-48, Default: 12
}
```

**Example Request:**
```javascript
const response = await fetch('/api/v1/nowcasting/batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    site_ids: ["KAMX", "KATX"],
    hours_back: 6
  })
});
```

**Response Schema:**
```typescript
interface BatchNowcastingResponse {
  success: boolean;
  predictions: Record<string, NowcastingPredictionResponse | {
    error: string;
    status: "failed";
  }>;
  total_sites: number;
  successful_sites: number;
  failed_sites: number;
  total_processing_time_ms: number;
  batch_timestamp: string;
}
```

**Usage:**
- **Multi-site weather dashboards**
- Compare predictions across different regions
- Handle partial failures gracefully

---

## 🗺️ **Raw Radar Data Endpoints (Frontend Integration)**

### 3. **GET** `/radar-data/{site_id}` - Get Raw Radar Data

**🎯 PRIMARY ENDPOINT FOR FRONTEND MAP INTEGRATION**

Returns raw radar data arrays with geographic coordinates for direct map rendering.

**Query Parameters:**
```typescript
interface RadarDataParams {
  hours_back?: number;                       // Default: 6, Max: 48
  max_frames?: number;                       // Default: 20, Max: 50
  include_processing_metadata?: boolean;     // Default: false
}
```

**Example Request:**
```javascript
const siteId = "KAMX";
const params = new URLSearchParams({
  hours_back: "6",
  max_frames: "10",
  include_processing_metadata: "false"
});

const response = await fetch(`/api/v1/nowcasting/radar-data/${siteId}?${params}`);
const data = await response.json();
```

**Response Schema:**
```typescript
interface RawRadarDataResponse {
  success: boolean;
  site_info: {
    site_id: string;
    name: string;
    location: string;
    coordinates: [number, number];           // [lat, lon]
    description: string;
  };
  frames: RadarDataFrame[];
  total_frames: number;
  time_range: {
    start: string;                          // ISO datetime
    end: string;                            // ISO datetime
  };
  processing_time_ms: number;
  cache_performance?: {
    cache_hits: number;
    cache_misses: number;
    processing_errors: string[];
  };
  request_timestamp: string;
}

interface RadarDataFrame {
  timestamp: string;                        // ISO datetime
  data: number[][];                         // 64x64 array of radar values (0-255)
  coordinates: {
    bounds: [number, number, number, number]; // [west, east, south, north]
    center: [number, number];               // [lat, lon]
    resolution_deg: number;                 // Degrees per pixel
    resolution_km: number;                  // Km per pixel
    projection: "PlateCarree";
    range_km: number;                       // Usually 150
  };
  intensity_range: [number, number];        // [min, max] values in data
  data_quality: "good" | "fair" | "poor";
  processing_metadata?: {
    file_path: string;
    file_size_bytes: number;
    coverage_ratio: number;
    non_zero_pixels: number;
    data_shape: [number, number];
  };
}
```

**Frontend Integration Example:**
```javascript
// Fetch radar data
const response = await fetch('/api/v1/nowcasting/radar-data/KAMX?max_frames=5');
const radarData = await response.json();

// Use with Leaflet
radarData.frames.forEach(frame => {
  const bounds = frame.coordinates.bounds;
  const leafletBounds = L.latLngBounds(
    [bounds[2], bounds[0]], // southwest
    [bounds[3], bounds[1]]  // northeast
  );
  
  // Create custom overlay
  const overlay = L.imageOverlay(
    convertDataToImageURL(frame.data), 
    leafletBounds
  ).addTo(map);
});

// Use with Mapbox GL JS
radarData.frames.forEach(frame => {
  map.addSource(`radar-${frame.timestamp}`, {
    type: 'image',
    url: convertDataToImageURL(frame.data),
    coordinates: [
      [frame.coordinates.bounds[0], frame.coordinates.bounds[3]], // top-left
      [frame.coordinates.bounds[1], frame.coordinates.bounds[3]], // top-right
      [frame.coordinates.bounds[1], frame.coordinates.bounds[2]], // bottom-right
      [frame.coordinates.bounds[0], frame.coordinates.bounds[2]]  // bottom-left
    ]
  });
});
```

---

### 4. **GET** `/radar-timeseries/{site_id}` - Get Radar Timeseries

Get radar data for a specific time range - perfect for historical analysis.

**Query Parameters:**
```typescript
interface TimeSeriesParams {
  start_time: string;                       // ISO datetime (required)
  end_time: string;                         // ISO datetime (required)
  max_frames?: number;                      // Default: 50, Max: 200
  include_processing_metadata?: boolean;    // Default: false
}
```

**Example Request:**
```javascript
const params = new URLSearchParams({
  start_time: "2025-08-02T16:00:00",
  end_time: "2025-08-02T19:00:00",
  max_frames: "20"
});

const response = await fetch(`/api/v1/nowcasting/radar-timeseries/KAMX?${params}`);
```

**Response Schema:**
```typescript
interface TimeSeriesRadarResponse {
  success: boolean;
  site_info: SiteInfo;
  frames: RadarDataFrame[];                 // Time-ordered frames
  total_frames: number;
  time_range: {                            // Requested range
    start: string;
    end: string;
  };
  actual_time_range: {                     // Actual data range
    start: string;
    end: string;
  };
  temporal_resolution_minutes: number;      // Average time between frames
  data_completeness: number;               // 0-1, percentage of time range covered
  processing_time_ms: number;
  request_timestamp: string;
}
```

**Usage:**
- **Historical weather analysis**
- **Time-lapse animations**
- **Weather event studies**

---

### 5. **GET** `/radar-data/multi-site` - Get Multi Site Radar Data

Get radar data from multiple sites for composite visualizations.

**Query Parameters:**
```typescript
interface MultiSiteParams {
  site_ids: string;                        // Comma-separated: "KAMX,KATX"
  hours_back?: number;                     // Default: 6
  max_frames_per_site?: number;            // Default: 10
}
```

**Example Request:**
```javascript
const params = new URLSearchParams({
  site_ids: "KAMX,KATX",
  hours_back: "6",
  max_frames_per_site: "5"
});

const response = await fetch(`/api/v1/nowcasting/radar-data/multi-site?${params}`);
```

**Response Schema:**
```typescript
interface MultiSiteRadarResponse {
  success: boolean;
  site_data: Record<string, RawRadarDataResponse>; // Keyed by site_id
  successful_sites: number;
  failed_sites: number;
  total_sites: number;
  composite_bounds?: {                     // Combined geographic bounds
    bounds: [number, number, number, number];
    center: [number, number];
    resolution_deg: number;
    resolution_km: number;
    projection: string;
    range_km: number;
    coverage_sites: string[];
  };
  processing_time_ms: number;
  request_timestamp: string;
}
```

**Usage:**
- **National/regional weather maps**
- **Multi-site radar composites**
- **Cross-regional weather analysis**

---

### 6. **GET** `/radar-frame/{site_id}` - Get Single Radar Frame

Get a single radar frame, either latest or for a specific timestamp.

**Query Parameters:**
```typescript
interface RadarFrameParams {
  timestamp?: string;                      // ISO datetime (optional, defaults to latest)
  include_processing_metadata?: boolean;   // Default: false
}
```

**Example Requests:**
```javascript
// Get latest frame
const latest = await fetch('/api/v1/nowcasting/radar-frame/KAMX');

// Get specific timestamp
const params = new URLSearchParams({
  timestamp: "2025-08-02T18:00:00",
  include_processing_metadata: "true"
});
const specific = await fetch(`/api/v1/nowcasting/radar-frame/KAMX?${params}`);
```

**Response Schema:**
```typescript
// Returns a single RadarDataFrame (same as in radar-data frames array)
interface RadarDataFrame {
  timestamp: string;
  data: number[][];                        // 64x64 array
  coordinates: CoordinateMetadata;
  intensity_range: [number, number];
  data_quality: "good" | "fair" | "poor";
  processing_metadata?: ProcessingMetadata;
}
```

**Usage:**
- **Real-time current conditions**
- **Specific moment analysis**
- **Quick data previews**

---

## 📊 **Status and Information Endpoints**

### 7. **GET** `/current-conditions/{site_id}` - Get Current Radar Conditions

Get current radar conditions and data availability.

**Response Schema:**
```typescript
interface CurrentRadarConditionsResponse {
  site_info: SiteInfo;
  latest_data_time?: string;               // ISO datetime
  data_freshness_hours?: number;           // Hours since latest data
  available_frames: number;
  data_quality: "excellent" | "good" | "fair" | "poor" | "no_data";
  coverage_area_km: number;                // Usually 150
  last_updated: string;                    // ISO datetime
}
```

**Usage:**
- **System health dashboards**
- **Data availability checks**
- **Quality monitoring**

---

### 8. **GET** `/sites` - Get Supported Sites

Get list of supported radar sites.

**Response Schema:**
```typescript
interface RadarSiteInfo {
  site_id: string;
  name: string;
  location: string;
  coordinates: [number, number];           // [lat, lon]
  description: string;
}

// Returns: RadarSiteInfo[]
```

**Usage:**
- **Site selection dropdowns**
- **Map initialization**
- **Feature discovery**

---

### 9. **GET** `/health` - Get Nowcasting Health

Get health status of the weather nowcasting system.

**Response Schema:**
```typescript
interface ModelHealthResponse {
  ml_model_name: string;
  is_loaded: boolean;
  ml_model_status: string;
  ml_model_info: Record<string, any>;
  health_check_time: string;
  last_prediction?: string;
  performance_metrics?: Record<string, number>;
}
```

---

### 10. **GET** `/data-status` - Get Data Pipeline Status

Get comprehensive status of data pipeline and processing services.

**Response Schema:**
```typescript
interface DataPipelineStatus {
  service_name: string;
  status: string;
  last_update: string;
  supported_sites: string[];
  site_status: Record<string, {
    directory_exists: boolean;
    available_dates: string[];
    total_files: number;
    total_size_mb: number;
  }>;
  storage_info: Record<string, any>;
  health_checks: Record<string, any>;
}
```

---

## 🔧 **Management Endpoints**

### 11. **POST** `/refresh-data` - Refresh Radar Data

Manually trigger refresh of radar data for specified sites.

**Request Schema:**
```typescript
interface DataRefreshRequest {
  site_ids?: string[];                     // Optional: specific sites, default: all
  hours_back?: number;                     // 1-24, Default: 6
  force_refresh?: boolean;                 // Default: false
}
```

**Response Schema:**
```typescript
interface DataRefreshResponse {
  success: boolean;
  sites_refreshed: string[];
  refresh_results: Record<string, any>;
  total_files_downloaded: number;
  total_processing_time_s: number;
  refresh_timestamp: string;
}
```

---

### 12. **POST** `/warm-cache` - Warm Cache For Site

Pre-load cache with recent radar data for faster predictions.

**Query Parameters:**
```typescript
interface WarmCacheParams {
  site_id: string;                         // Required in URL path
  hours_back?: number;                     // Default: 6
}
```

---

### 13. **GET** `/cache-stats` - Get Cache Statistics

Get cache performance statistics.

**Response Schema:**
```typescript
interface CacheStats {
  service: string;
  cache_enabled: boolean;
  cache_stats?: {
    hits: number;
    misses: number;
    hit_rate: number;
  };
}
```

---

## 🖼️ **Visualization Endpoints (Legacy - Use Raw Data Instead)**

### 14. **GET** `/visualization/{site_id}` - Get Radar Visualization

⚠️ **Deprecated for frontend use** - Returns PNG image instead of data arrays.

**Response:** PNG image file

**Usage:** Use `/radar-data/{site_id}` instead for frontend integration.

---

### 15. **GET** `/mosaic` - Get Radar Mosaic

⚠️ **Deprecated for frontend use** - Returns composite PNG image.

**Response:** PNG image file

**Usage:** Use `/radar-data/multi-site` instead for frontend integration.

---

### 16. **GET** `/mosaic/info` - Get Mosaic Info

Get information about radar mosaic service capabilities.

**Response Schema:**
```typescript
interface MosaicServiceInfo {
  service: string;
  supported_sites: string[];
  site_details: Record<string, SiteInfo>;
  capabilities: string[];
  output_formats: string[];
  projections: string[];
  colormap: string;
}
```

---

## 🚀 **Frontend Integration Best Practices**

### **For Real-Time Weather Maps:**
1. Use `/radar-data/{site_id}` for current conditions
2. Use `/radar-frame/{site_id}` for quick updates
3. Implement polling every 5-10 minutes for live data

### **For Historical Analysis:**
1. Use `/radar-timeseries/{site_id}` with specific date ranges
2. Set appropriate `max_frames` to avoid large responses
3. Check `data_completeness` for data quality assessment

### **For Multi-Site Dashboards:**
1. Use `/radar-data/multi-site` for composite views
2. Handle partial failures gracefully
3. Use `composite_bounds` for map centering

### **Performance Tips:**
1. Enable caching with `/warm-cache` for frequently accessed sites
2. Use `max_frames` parameter to limit response size
3. Monitor `/cache-stats` for performance optimization

### **Error Handling:**
```javascript
const response = await fetch('/api/v1/nowcasting/radar-data/KAMX');

if (!response.ok) {
  const error = await response.json();
  console.error('API Error:', error.detail);
  return;
}

const data = await response.json();
if (!data.success) {
  console.error('Request failed:', data);
  return;
}

// Process successful response
processRadarData(data);
```

### **TypeScript Integration:**
```typescript
// Define types
import type { 
  RawRadarDataResponse, 
  RadarDataFrame, 
  TimeSeriesRadarResponse 
} from './types/nowcasting';

// Use with proper typing
const fetchRadarData = async (siteId: string): Promise<RawRadarDataResponse | null> => {
  try {
    const response = await fetch(`/api/v1/nowcasting/radar-data/${siteId}`);
    if (!response.ok) return null;
    
    const data: RawRadarDataResponse = await response.json();
    return data.success ? data : null;
  } catch (error) {
    console.error('Failed to fetch radar data:', error);
    return null;
  }
};
```

## 🗺️ **Creating Professional Weather Radar Visualizations**

### **Converting Raw Data to Geographic Weather Maps**

The radar data from the API comes as 64x64 arrays with coordinate metadata. Here's how to render it as professional weather maps with geographic context like the NWS radar displays:

#### **Method 1: Canvas-Based Rendering (Recommended)**

This approach creates a weather radar image similar to professional weather services:

```javascript
/**
 * Convert radar data to a weather map image with geographic context
 */
class WeatherRadarRenderer {
  constructor(canvasId, width = 800, height = 600) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.canvas.width = width;
    this.canvas.height = height;
    
    // NWS-style color scheme for reflectivity (dBZ)
    this.colorScale = [
      { dbz: -32, color: [0, 0, 0, 0] },        // Transparent (no data)
      { dbz: -20, color: [156, 156, 156, 180] }, // Light gray
      { dbz: -10, color: [118, 118, 118, 180] }, // Medium gray  
      { dbz: 0,   color: [0, 236, 236, 200] },   // Light blue
      { dbz: 10,  color: [1, 160, 246, 220] },   // Blue
      { dbz: 20,  color: [0, 0, 246, 240] },     // Dark blue
      { dbz: 30,  color: [0, 255, 0, 240] },     // Green
      { dbz: 40,  color: [0, 187, 0, 240] },     // Dark green
      { dbz: 45,  color: [255, 255, 0, 240] },   // Yellow
      { dbz: 50,  color: [255, 216, 0, 240] },   // Gold
      { dbz: 55,  color: [255, 149, 0, 240] },   // Orange
      { dbz: 60,  color: [255, 0, 0, 240] },     // Red
      { dbz: 65,  color: [214, 0, 0, 240] },     // Dark red
      { dbz: 70,  color: [192, 0, 0, 240] },     // Maroon
      { dbz: 75,  color: [255, 0, 255, 240] },   // Magenta
      { dbz: 80,  color: [153, 85, 196, 240] }   // Purple
    ];
  }

  /**
   * Convert 0-255 radar value to dBZ (reflectivity)
   */
  valueToDbz(value) {
    // API returns 0-255, convert to -32 to +95 dBZ range
    return (value / 255.0) * 127 - 32;
  }

  /**
   * Get color for a dBZ value using NWS color scheme
   */
  getColorForDbz(dbz) {
    for (let i = this.colorScale.length - 1; i >= 0; i--) {
      if (dbz >= this.colorScale[i].dbz) {
        return this.colorScale[i].color;
      }
    }
    return [0, 0, 0, 0]; // Transparent for values below range
  }

  /**
   * Render radar data with geographic background
   */
  renderRadarData(radarFrame, showGeography = true) {
    const { data, coordinates } = radarFrame;
    
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Step 1: Draw geographic background (states, coastlines)
    if (showGeography) {
      this.drawGeographicBackground(coordinates);
    }
    
    // Step 2: Create radar data image
    const radarImageData = this.createRadarImageData(data);
    
    // Step 3: Draw radar overlay with proper transparency
    this.drawRadarOverlay(radarImageData, coordinates);
    
    // Step 4: Add range rings and site marker
    this.drawRadarSite(coordinates);
    
    // Step 5: Add scale and labels
    this.drawColorScale();
    this.drawLabels(radarFrame);
  }

  /**
   * Create ImageData from radar array with NWS colors
   */
  createRadarImageData(radarData) {
    const width = radarData.length;
    const height = radarData[0].length;
    const imageData = this.ctx.createImageData(width, height);
    const data = imageData.data;
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIndex = (y * width + x) * 4;
        const radarValue = radarData[y][x];
        const dbz = this.valueToDbz(radarValue);
        const [r, g, b, a] = this.getColorForDbz(dbz);
        
        data[pixelIndex] = r;     // Red
        data[pixelIndex + 1] = g; // Green  
        data[pixelIndex + 2] = b; // Blue
        data[pixelIndex + 3] = a; // Alpha
      }
    }
    
    return imageData;
  }

  /**
   * Draw geographic background (states, coastlines)
   */
  drawGeographicBackground(coordinates) {
    const bounds = coordinates.bounds; // [west, east, south, north]
    
    // Set background color
    this.ctx.fillStyle = '#f0f8ff'; // Light blue background
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw state boundaries (you'll need to load state boundary data)
    this.drawStateBoundaries(bounds);
    
    // Draw coastlines
    this.drawCoastlines(bounds);
    
    // Add subtle grid
    this.drawGrid(bounds);
  }

  /**
   * Draw state boundaries - you'll need GeoJSON data for this
   */
  drawStateBoundaries(bounds) {
    // This requires loading US state boundary GeoJSON data
    // Example using a hypothetical state boundaries dataset:
    
    this.ctx.strokeStyle = '#666666';
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([2, 2]);
    
    // You would load and render actual state boundary data here
    // For now, draw a simple example for Florida/Miami area
    if (bounds[0] < -80 && bounds[1] > -80) { // Miami area
      this.drawFloridaBoundary(bounds);
    }
    
    this.ctx.setLineDash([]); // Reset line dash
  }

  /**
   * Draw coastlines
   */
  drawCoastlines(bounds) {
    this.ctx.strokeStyle = '#333333';
    this.ctx.lineWidth = 2;
    
    // You would load and render actual coastline data here
    // This is a simplified example
  }

  /**
   * Draw coordinate grid
   */
  drawGrid(bounds) {
    this.ctx.strokeStyle = '#cccccc';
    this.ctx.lineWidth = 0.5;
    this.ctx.setLineDash([1, 3]);
    
    // Draw latitude lines
    const latStep = Math.ceil((bounds[3] - bounds[2]) / 5); // ~5 lines
    for (let lat = Math.ceil(bounds[2]); lat <= Math.floor(bounds[3]); lat += latStep) {
      const y = this.latLonToPixel(lat, bounds[0], bounds).y;
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(this.canvas.width, y);
      this.ctx.stroke();
    }
    
    // Draw longitude lines  
    const lonStep = Math.ceil((bounds[1] - bounds[0]) / 5); // ~5 lines
    for (let lon = Math.ceil(bounds[0]); lon <= Math.floor(bounds[1]); lon += lonStep) {
      const x = this.latLonToPixel(bounds[2], lon, bounds).x;
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, this.canvas.height);
      this.ctx.stroke();
    }
    
    this.ctx.setLineDash([]);
  }

  /**
   * Convert lat/lon to canvas pixel coordinates
   */
  latLonToPixel(lat, lon, bounds) {
    const x = ((lon - bounds[0]) / (bounds[1] - bounds[0])) * this.canvas.width;
    const y = ((bounds[3] - lat) / (bounds[3] - bounds[2])) * this.canvas.height;
    return { x, y };
  }

  /**
   * Draw radar overlay on geographic background
   */
  drawRadarOverlay(radarImageData, coordinates) {
    // Create temporary canvas for radar data
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 64;
    tempCanvas.height = 64;
    
    // Put radar data on temp canvas
    tempCtx.putImageData(radarImageData, 0, 0);
    
    // Draw scaled radar data on main canvas
    this.ctx.globalAlpha = 0.7; // Semi-transparent overlay
    this.ctx.drawImage(
      tempCanvas, 
      0, 0, 64, 64,                           // Source
      0, 0, this.canvas.width, this.canvas.height // Destination (scaled)
    );
    this.ctx.globalAlpha = 1.0; // Reset transparency
  }

  /**
   * Draw radar site marker and range rings
   */
  drawRadarSite(coordinates) {
    const center = coordinates.center; // [lat, lon]
    const bounds = coordinates.bounds;
    const sitePixel = this.latLonToPixel(center[0], center[1], bounds);
    
    // Draw range rings (25km, 50km, 100km, 150km)
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([5, 5]);
    
    const rangeRings = [25, 50, 100, 150]; // km
    rangeRings.forEach(range => {
      const radiusPixels = (range / 150) * (this.canvas.width / 2); // Approximate
      this.ctx.beginPath();
      this.ctx.arc(sitePixel.x, sitePixel.y, radiusPixels, 0, 2 * Math.PI);
      this.ctx.stroke();
    });
    
    this.ctx.setLineDash([]);
    
    // Draw radar site marker
    this.ctx.fillStyle = '#ff0000';
    this.ctx.beginPath();
    this.ctx.arc(sitePixel.x, sitePixel.y, 4, 0, 2 * Math.PI);
    this.ctx.fill();
    
    // Label the site
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '12px Arial';
    this.ctx.fillText('📡', sitePixel.x + 8, sitePixel.y + 4);
  }

  /**
   * Draw reflectivity color scale
   */
  drawColorScale() {
    const scaleX = this.canvas.width - 120;
    const scaleY = 50;
    const scaleWidth = 20;
    const scaleHeight = 200;
    
    // Draw scale background
    this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    this.ctx.fillRect(scaleX - 5, scaleY - 20, scaleWidth + 60, scaleHeight + 40);
    
    // Draw color gradient
    const gradient = this.ctx.createLinearGradient(0, scaleY, 0, scaleY + scaleHeight);
    this.colorScale.forEach((stop, i) => {
      const position = i / (this.colorScale.length - 1);
      const [r, g, b, a] = stop.color;
      gradient.addColorStop(position, `rgba(${r},${g},${b},${a/255})`);
    });
    
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(scaleX, scaleY, scaleWidth, scaleHeight);
    
    // Draw scale labels
    this.ctx.fillStyle = '#000000';
    this.ctx.font = '10px Arial';
    this.ctx.textAlign = 'left';
    
    [80, 60, 40, 20, 0, -20].forEach((dbz, i) => {
      const y = scaleY + (i / 5) * scaleHeight;
      this.ctx.fillText(`${dbz}`, scaleX + 25, y + 3);
    });
    
    // Scale title
    this.ctx.font = '12px Arial';
    this.ctx.fillText('dBZ', scaleX, scaleY - 5);
  }

  /**
   * Draw labels and info
   */
  drawLabels(radarFrame) {
    // Site info
    this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    this.ctx.fillRect(10, 10, 200, 80);
    
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '14px Arial';
    this.ctx.textAlign = 'left';
    
    const timestamp = new Date(radarFrame.timestamp);
    const quality = radarFrame.data_quality.toUpperCase();
    
    this.ctx.fillText('NEXRAD Radar', 15, 30);
    this.ctx.fillText(`Quality: ${quality}`, 15, 50);
    this.ctx.fillText(`Time: ${timestamp.toLocaleString()}`, 15, 70);
    
    // North arrow
    this.drawNorthArrow();
  }

  /**
   * Draw north arrow
   */
  drawNorthArrow() {
    const x = this.canvas.width - 50;
    const y = 50;
    
    this.ctx.strokeStyle = '#000000';
    this.ctx.fillStyle = '#ffffff';
    this.ctx.lineWidth = 2;
    
    // Arrow
    this.ctx.beginPath();
    this.ctx.moveTo(x, y - 15);
    this.ctx.lineTo(x - 8, y + 5);
    this.ctx.lineTo(x, y);
    this.ctx.lineTo(x + 8, y + 5);
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.stroke();
    
    // N label
    this.ctx.fillStyle = '#000000';
    this.ctx.font = '12px Arial';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('N', x, y + 25);
  }
}
```

#### **Method 2: Leaflet Integration with Custom Overlay**

For interactive maps with zoom/pan capabilities:

```javascript
/**
 * Leaflet-based radar overlay
 */
class LeafletRadarOverlay {
  constructor(mapId) {
    // Initialize map with geographic context
    this.map = L.map(mapId, {
      center: [25.6112, -80.4128], // Miami default
      zoom: 8
    });
    
    // Add base map with states/coastlines
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(this.map);
    
    // Add US state boundaries
    this.addStateBoundaries();
  }

  async addStateBoundaries() {
    // Load US states GeoJSON (you can get this from various sources)
    try {
      const response = await fetch('/data/us-states.geojson');
      const statesData = await response.json();
      
      L.geoJSON(statesData, {
        style: {
          color: '#666666',
          weight: 1,
          fillOpacity: 0
        }
      }).addTo(this.map);
    } catch (error) {
      console.warn('Could not load state boundaries:', error);
    }
  }

  addRadarData(radarFrame) {
    const bounds = radarFrame.coordinates.bounds;
    const leafletBounds = L.latLngBounds(
      [bounds[2], bounds[0]], // southwest
      [bounds[3], bounds[1]]  // northeast
    );
    
    // Convert radar data to image URL
    const imageUrl = this.createRadarImageUrl(radarFrame.data);
    
    // Add radar overlay
    const radarOverlay = L.imageOverlay(imageUrl, leafletBounds, {
      opacity: 0.7,
      interactive: false
    }).addTo(this.map);
    
    // Add radar site marker
    const siteMarker = L.marker(radarFrame.coordinates.center, {
      icon: L.divIcon({
        html: '📡',
        className: 'radar-site-marker',
        iconSize: [20, 20]
      })
    }).addTo(this.map);
    
    // Fit map to radar bounds
    this.map.fitBounds(leafletBounds);
    
    return radarOverlay;
  }

  createRadarImageUrl(radarData) {
    // Create canvas and convert to data URL
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.height = 64;
    
    const imageData = ctx.createImageData(64, 64);
    const data = imageData.data;
    
    // Apply NWS color scheme (same as above)
    for (let y = 0; y < 64; y++) {
      for (let x = 0; x < 64; x++) {
        const pixelIndex = (y * 64 + x) * 4;
        const radarValue = radarData[y][x];
        const dbz = (radarValue / 255.0) * 127 - 32;
        const [r, g, b, a] = this.getColorForDbz(dbz);
        
        data[pixelIndex] = r;
        data[pixelIndex + 1] = g;
        data[pixelIndex + 2] = b;
        data[pixelIndex + 3] = a;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  }

  // Include the same getColorForDbz method from above
  getColorForDbz(dbz) {
    const colorScale = [
      { dbz: -32, color: [0, 0, 0, 0] },
      { dbz: -20, color: [156, 156, 156, 180] },
      { dbz: -10, color: [118, 118, 118, 180] },
      { dbz: 0,   color: [0, 236, 236, 200] },
      { dbz: 10,  color: [1, 160, 246, 220] },
      { dbz: 20,  color: [0, 0, 246, 240] },
      { dbz: 30,  color: [0, 255, 0, 240] },
      { dbz: 40,  color: [0, 187, 0, 240] },
      { dbz: 45,  color: [255, 255, 0, 240] },
      { dbz: 50,  color: [255, 216, 0, 240] },
      { dbz: 55,  color: [255, 149, 0, 240] },
      { dbz: 60,  color: [255, 0, 0, 240] },
      { dbz: 65,  color: [214, 0, 0, 240] },
      { dbz: 70,  color: [192, 0, 0, 240] },
      { dbz: 75,  color: [255, 0, 255, 240] },
      { dbz: 80,  color: [153, 85, 196, 240] }
    ];
    
    for (let i = colorScale.length - 1; i >= 0; i--) {
      if (dbz >= colorScale[i].dbz) {
        return colorScale[i].color;
      }
    }
    return [0, 0, 0, 0];
  }
}
```

#### **Usage Examples:**

```javascript
// Method 1: Canvas rendering
const renderer = new WeatherRadarRenderer('radar-canvas', 800, 600);

// Fetch and render radar data
const response = await fetch('/api/v1/nowcasting/radar-data/KAMX?max_frames=1');
const radarData = await response.json();

if (radarData.success && radarData.frames.length > 0) {
  renderer.renderRadarData(radarData.frames[0], true);
}

// Method 2: Leaflet integration
const leafletRadar = new LeafletRadarOverlay('radar-map');
leafletRadar.addRadarData(radarData.frames[0]);
```

#### **Required Geographic Data:**

To properly render the geographic background, you'll need:

1. **US State Boundaries GeoJSON** - Available from:
   - https://github.com/holtzy/D3-graph-gallery/blob/master/DATA/world.geojson
   - https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json

2. **Coastline Data** - Can be extracted from OpenStreetMap or Natural Earth data

3. **Base Map Tiles** (for Leaflet approach):
   - OpenStreetMap (free)
   - Mapbox (requires API key)
   - CartoDB (free tier available)

This approach will give you professional weather radar visualizations similar to the NWS displays, with proper geographic context including state boundaries, coastlines, and coordinate grids.

---

This guide provides everything needed for robust frontend integration with the SkyGuard Analytics Nowcasting API. Focus on the raw radar data endpoints (`/radar-data/*`) for modern web applications that need direct data manipulation and custom visualizations.