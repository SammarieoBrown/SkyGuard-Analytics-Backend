# SkyGuard Analytics Nowcasting API - Frontend Integration Guide

## LLM Agent Instructions

You are tasked with integrating the SkyGuard Analytics Nowcasting APIs into a Next.js/React frontend application. These APIs provide real-time weather radar data as **raw arrays with geographic coordinates** - NOT as images. You will need to render this data on interactive maps using libraries like Leaflet, Mapbox, or similar mapping solutions.

### Key Principles:
1. **Never request image endpoints** - All radar visualizations should be built client-side using the raw data arrays
2. **Use coordinate metadata** to properly position radar data on maps
3. **Implement client-side rendering** for radar overlays, heatmaps, and mosaics
4. **Cache aggressively** - Radar data updates slowly (every 5-10 minutes)
5. **Handle multiple sites** for composite/mosaic views across regions

## Base Configuration

```typescript
const API_BASE_URL = 'https://skyguard-analytics-backend.onrender.com';
const API_VERSION = '/api/v1';
const NOWCASTING_BASE = `${API_BASE_URL}${API_VERSION}/nowcasting`;

// Available radar sites
const RADAR_SITES = {
  KAMX: { name: 'Miami', lat: 25.6111, lon: -80.4128 },
  KATX: { name: 'Seattle', lat: 48.1945, lon: -122.4958 }
};
```

## API Endpoints Documentation

### 1. POST `/api/v1/nowcasting/predict` - Predict Weather Nowcast

**Purpose**: Generate 6-frame future weather predictions using ML models based on current radar data.

**Request Schema**:
```typescript
interface NowcastingPredictionRequest {
  site_id: 'KAMX' | 'KATX';  // Required: Radar site identifier
  use_latest_data: boolean;   // Default: true - Use latest radar data
  hours_back: number;         // Default: 12, Range: 1-48 hours
  custom_radar_data?: number[][][][];  // Optional: Custom input (shape: [10, 64, 64, 1])
}
```

**Response Schema**:
```typescript
interface NowcastingPredictionResponse {
  success: boolean;
  site_info: RadarSiteInfo;
  prediction_frames: number[][][][];  // Shape: [6, 64, 64, 1] - 6 future frames
  input_metadata: {
    data_source: string;
    files_used?: number;
    processing_metadata?: any;
  };
  ml_model_metadata: Record<string, any>;
  processing_time_ms: number;
  prediction_timestamp: string;  // ISO datetime
  frame_times: string[];  // Predicted time for each frame
  confidence_metrics?: Record<string, number>;
}
```

**Frontend Usage**:
```typescript
async function getPrediction(siteId: string) {
  const response = await fetch(`${NOWCASTING_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      site_id: siteId,
      use_latest_data: true,
      hours_back: 6
    })
  });
  
  const data = await response.json();
  // Each prediction frame is a 64x64 array representing future precipitation
  // Render these on your map as animated overlays
  return data;
}
```

### 2. POST `/api/v1/nowcasting/batch` - Batch Weather Nowcast

**Purpose**: Get predictions for multiple radar sites simultaneously.

**Request Schema**:
```typescript
interface BatchNowcastingRequest {
  site_ids: string[];  // Array of site IDs: ['KAMX', 'KATX']
  hours_back: number;  // Default: 12, Range: 1-48
}
```

**Response Schema**:
```typescript
interface BatchNowcastingResponse {
  success: boolean;
  predictions: Record<string, NowcastingPredictionResponse | ErrorInfo>;
  total_sites: number;
  successful_sites: number;
  failed_sites: number;
  total_processing_time_ms: number;
  batch_timestamp: string;
}
```

### 3. GET `/api/v1/nowcasting/current-conditions/{site_id}` - Get Current Radar Conditions

**Purpose**: Check data availability and quality for a specific radar site.

**Response Schema**:
```typescript
interface CurrentRadarConditionsResponse {
  site_info: RadarSiteInfo;
  latest_data_time: string | null;  // ISO datetime
  data_freshness_hours: number | null;
  available_frames: number;
  data_quality: 'excellent' | 'good' | 'fair' | 'poor' | 'no_data';
  coverage_area_km: number;  // Default: 150
  last_updated: string;
}
```

### 4. GET `/api/v1/nowcasting/sites` - Get Supported Sites

**Purpose**: List all available radar sites with their metadata.

**Response Schema**:
```typescript
interface RadarSiteInfo {
  site_id: string;
  name: string;
  location: string;
  coordinates: [number, number];  // [latitude, longitude]
  description: string;
}

// Returns: RadarSiteInfo[]
```

### 5. GET `/api/v1/nowcasting/health` - Get Nowcasting Health

**Purpose**: Check ML model and system health status.

**Response Schema**:
```typescript
interface ModelHealthResponse {
  ml_model_name: string;
  is_loaded: boolean;
  ml_model_status: string;
  last_prediction?: string;
  ml_model_info: Record<string, any>;
  performance_metrics?: Record<string, number>;
  health_check_time: string;
}
```

### 6. GET `/api/v1/nowcasting/data-status` - Get Data Pipeline Status

**Purpose**: Monitor data pipeline and processing service status.

**Response Schema**:
```typescript
interface DataPipelineStatus {
  service_name: string;
  status: string;
  last_update: string;
  supported_sites: string[];
  site_status: Record<string, Record<string, any>>;
  storage_info: Record<string, any>;
  health_checks: Record<string, any>;
}
```

### 7. POST `/api/v1/nowcasting/refresh-data` - Refresh Radar Data

**Purpose**: Manually trigger data refresh for specified sites.

**Request Schema**:
```typescript
interface DataRefreshRequest {
  site_ids?: string[];  // Optional: specific sites (default: all)
  hours_back: number;   // Default: 6, Range: 1-24
  force_refresh: boolean;  // Default: false
}
```

### 8. POST `/api/v1/nowcasting/warm-cache` - Warm Cache For Site

**Purpose**: Pre-load radar data into cache for faster access.

**Query Parameters**:
- `site_id` (required): Radar site identifier
- `hours_back` (optional): Hours of data to cache (default: 6)

### 9. GET `/api/v1/nowcasting/cache-stats` - Get Cache Statistics

**Purpose**: Monitor cache performance and hit rates.

**Response**: Returns cache statistics including hit/miss ratios, memory usage, etc.

### 10. GET `/api/v1/nowcasting/visualization/{site_id}` - Get Radar Visualization

**⚠️ AVOID THIS ENDPOINT** - Returns server-rendered PNG image. Use raw data endpoints instead for frontend rendering.

### 11. GET `/api/v1/nowcasting/mosaic` - Get Radar Mosaic

**⚠️ AVOID THIS ENDPOINT** - Returns server-rendered PNG mosaic. Use multi-site raw data instead.

### 12. GET `/api/v1/nowcasting/mosaic/info` - Get Mosaic Info

**Purpose**: Get information about mosaic service capabilities.

**Response**: Service configuration and supported features.

### 13. GET `/api/v1/nowcasting/radar-data/multi-site` - Get Multi Site Radar Data

**🔥 CRITICAL FOR FRONTEND** - Use this for composite visualizations!

**Purpose**: Get raw radar data from multiple sites for creating client-side mosaics.

**Query Parameters**:
- `site_ids`: Comma-separated site IDs (e.g., "KAMX,KATX")
- `hours_back`: Hours of historical data (default: 6)
- `max_frames_per_site`: Maximum frames per site (default: 10)

**Response Schema**:
```typescript
interface MultiSiteRadarResponse {
  success: boolean;
  site_data: Record<string, RawRadarDataResponse>;
  successful_sites: number;
  failed_sites: number;
  total_sites: number;
  composite_bounds?: CoordinateMetadata;  // Combined geographic bounds
  processing_time_ms: number;
  request_timestamp: string;
}
```

**Frontend Implementation**:
```typescript
async function createRadarMosaic() {
  const response = await fetch(
    `${NOWCASTING_BASE}/radar-data/multi-site?site_ids=KAMX,KATX&max_frames_per_site=1`
  );
  const data = await response.json();
  
  // Render each site's data on the map
  Object.entries(data.site_data).forEach(([siteId, siteData]) => {
    if (siteData.success && siteData.frames.length > 0) {
      const frame = siteData.frames[0];
      renderRadarOverlay(map, frame.data, frame.coordinates);
    }
  });
}
```

### 14. GET `/api/v1/nowcasting/radar-data/{site_id}` - Get Raw Radar Data

**🔥 PRIMARY DATA ENDPOINT** - Essential for frontend radar displays!

**Purpose**: Get processed radar data arrays with geographic coordinates.

**Query Parameters**:
- `hours_back`: Historical data range (default: 6)
- `max_frames`: Maximum frames to return (default: 20)
- `include_processing_metadata`: Include detailed metadata (default: false)

**Response Schema**:
```typescript
interface RawRadarDataResponse {
  success: boolean;
  site_info: RadarSiteInfo;
  frames: RadarDataFrame[];  // Time-ordered radar frames
  total_frames: number;
  time_range: {
    start: string;  // ISO datetime
    end: string;
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
  timestamp: string;  // ISO datetime
  data: number[][];   // 64x64 array of intensity values
  coordinates: CoordinateMetadata;
  intensity_range: [number, number];  // [min, max]
  data_quality: 'good' | 'fair' | 'poor';
  processing_metadata?: Record<string, any>;
}

interface CoordinateMetadata {
  bounds: [number, number, number, number];  // [west, east, south, north] in degrees
  center: [number, number];  // [latitude, longitude]
  resolution_deg: number;  // Degrees per pixel
  resolution_km: number;   // Kilometers per pixel
  projection: string;      // Default: "PlateCarree"
  range_km: number;       // Radar range (default: 150)
}
```

**Frontend Rendering Example**:
```typescript
import L from 'leaflet';

async function displayRadarData(siteId: string, map: L.Map) {
  const response = await fetch(
    `${NOWCASTING_BASE}/radar-data/${siteId}?hours_back=1&max_frames=5`
  );
  const data: RawRadarDataResponse = await response.json();
  
  if (!data.success || data.frames.length === 0) return;
  
  // Process each frame
  data.frames.forEach((frame, index) => {
    const { data: radarData, coordinates } = frame;
    
    // Create canvas for rendering
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    
    // Render radar data to canvas
    const imageData = ctx.createImageData(64, 64);
    for (let y = 0; y < 64; y++) {
      for (let x = 0; x < 64; x++) {
        const value = radarData[y][x];
        const intensity = Math.floor((value / frame.intensity_range[1]) * 255);
        const idx = (y * 64 + x) * 4;
        
        // Color mapping for precipitation (blue gradient)
        imageData.data[idx] = 0;      // R
        imageData.data[idx + 1] = intensity; // G  
        imageData.data[idx + 2] = 255; // B
        imageData.data[idx + 3] = intensity > 10 ? intensity : 0; // Alpha
      }
    }
    ctx.putImageData(imageData, 0, 0);
    
    // Add as map overlay
    const imageBounds = L.latLngBounds(
      [coordinates.bounds[2], coordinates.bounds[0]], // SW
      [coordinates.bounds[3], coordinates.bounds[1]]  // NE
    );
    
    L.imageOverlay(canvas.toDataURL(), imageBounds, {
      opacity: 0.7,
      className: `radar-frame-${index}`
    }).addTo(map);
  });
}
```

### 15. GET `/api/v1/nowcasting/radar-timeseries/{site_id}` - Get Radar Timeseries

**Purpose**: Get historical radar data for a specific time range.

**Query Parameters**:
- `start_time`: ISO datetime string (required)
- `end_time`: ISO datetime string (required)
- `max_frames`: Maximum frames to return (default: 50)
- `include_processing_metadata`: Include metadata (default: false)

**Response Schema**:
```typescript
interface TimeSeriesRadarResponse {
  success: boolean;
  site_info: RadarSiteInfo;
  frames: RadarDataFrame[];
  total_frames: number;
  time_range: { start: string; end: string };  // Requested range
  actual_time_range: { start: string; end: string };  // Actual data range
  temporal_resolution_minutes: number;  // Average time between frames
  data_completeness: number;  // 0-1, percentage of requested range covered
  processing_time_ms: number;
  request_timestamp: string;
}
```

**Usage Example**:
```typescript
async function getHistoricalRadar(siteId: string, hours: number) {
  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - hours * 60 * 60 * 1000);
  
  const response = await fetch(
    `${NOWCASTING_BASE}/radar-timeseries/${siteId}?` +
    `start_time=${startTime.toISOString()}&` +
    `end_time=${endTime.toISOString()}&` +
    `max_frames=30`
  );
  
  return response.json();
}
```

### 16. GET `/api/v1/nowcasting/radar-frame/{site_id}` - Get Single Radar Frame

**Purpose**: Get a single radar frame (latest or specific timestamp).

**Query Parameters**:
- `timestamp`: ISO datetime string (optional, default: latest)
- `include_processing_metadata`: Include metadata (default: false)

**Response**: Returns a single `RadarDataFrame` object.

## Frontend Rendering Strategies

### 1. Canvas-Based Rendering

```typescript
function renderRadarToCanvas(
  radarData: number[][],
  colorMap: 'precipitation' | 'velocity' | 'reflectivity'
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(64, 64);
  
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const value = radarData[y][x];
      const color = getColorForValue(value, colorMap);
      const idx = (y * 64 + x) * 4;
      
      imageData.data[idx] = color.r;
      imageData.data[idx + 1] = color.g;
      imageData.data[idx + 2] = color.b;
      imageData.data[idx + 3] = color.a;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function getColorForValue(
  value: number,
  colorMap: string
): { r: number; g: number; b: number; a: number } {
  // Implement color mapping based on precipitation intensity
  if (colorMap === 'precipitation') {
    if (value < 0.1) return { r: 0, g: 0, b: 0, a: 0 };
    if (value < 0.3) return { r: 0, g: 100, b: 255, a: 128 };
    if (value < 0.5) return { r: 0, g: 200, b: 255, a: 180 };
    if (value < 0.7) return { r: 255, g: 255, b: 0, a: 200 };
    return { r: 255, g: 0, b: 0, a: 255 };
  }
  // Add other color maps...
  return { r: 0, g: 0, b: 0, a: 0 };
}
```

### 2. WebGL Rendering (High Performance)

```typescript
import * as THREE from 'three';

function createRadarMesh(
  radarData: number[][],
  coordinates: CoordinateMetadata
): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(
    coordinates.range_km * 2,
    coordinates.range_km * 2,
    63, 63
  );
  
  const vertices = geometry.attributes.position.array;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const idx = (y * 64 + x) * 3 + 2; // Z coordinate
      vertices[idx] = radarData[y][x] * 10; // Scale height
    }
  }
  
  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.7
  });
  
  return new THREE.Mesh(geometry, material);
}
```

### 3. Animated Time Series

```typescript
class RadarAnimation {
  private frames: RadarDataFrame[];
  private currentFrame: number = 0;
  private animationId: number | null = null;
  
  constructor(frames: RadarDataFrame[]) {
    this.frames = frames;
  }
  
  play(map: L.Map, fps: number = 5) {
    const frameDuration = 1000 / fps;
    
    const animate = () => {
      this.renderFrame(map, this.frames[this.currentFrame]);
      this.currentFrame = (this.currentFrame + 1) % this.frames.length;
      this.animationId = setTimeout(animate, frameDuration);
    };
    
    animate();
  }
  
  stop() {
    if (this.animationId) {
      clearTimeout(this.animationId);
      this.animationId = null;
    }
  }
  
  private renderFrame(map: L.Map, frame: RadarDataFrame) {
    // Clear previous overlay
    map.eachLayer(layer => {
      if (layer.options?.className?.startsWith('radar-overlay')) {
        map.removeLayer(layer);
      }
    });
    
    // Render new frame
    const canvas = renderRadarToCanvas(frame.data, 'precipitation');
    const bounds = L.latLngBounds(
      [frame.coordinates.bounds[2], frame.coordinates.bounds[0]],
      [frame.coordinates.bounds[3], frame.coordinates.bounds[1]]
    );
    
    L.imageOverlay(canvas.toDataURL(), bounds, {
      opacity: 0.7,
      className: 'radar-overlay'
    }).addTo(map);
  }
}
```

### 4. Creating Client-Side Mosaics

```typescript
async function createCompositeMosaic(map: L.Map) {
  // Fetch data from multiple sites
  const response = await fetch(
    `${NOWCASTING_BASE}/radar-data/multi-site?site_ids=KAMX,KATX`
  );
  const data: MultiSiteRadarResponse = await response.json();
  
  // Create composite canvas
  const compositeCanvas = document.createElement('canvas');
  compositeCanvas.width = 256;  // Larger for mosaic
  compositeCanvas.height = 256;
  const ctx = compositeCanvas.getContext('2d');
  
  // Blend data from multiple sites
  Object.entries(data.site_data).forEach(([siteId, siteData]) => {
    if (siteData.success && siteData.frames.length > 0) {
      const frame = siteData.frames[0];
      const siteCanvas = renderRadarToCanvas(frame.data, 'precipitation');
      
      // Calculate position in mosaic based on coordinates
      const x = calculateMosaicX(frame.coordinates.center[1]);
      const y = calculateMosaicY(frame.coordinates.center[0]);
      
      // Draw with blending
      ctx.globalCompositeOperation = 'screen';
      ctx.drawImage(siteCanvas, x, y, 128, 128);
    }
  });
  
  // Add composite to map
  if (data.composite_bounds) {
    const bounds = L.latLngBounds(
      [data.composite_bounds.bounds[2], data.composite_bounds.bounds[0]],
      [data.composite_bounds.bounds[3], data.composite_bounds.bounds[1]]
    );
    
    L.imageOverlay(compositeCanvas.toDataURL(), bounds, {
      opacity: 0.8,
      className: 'radar-mosaic'
    }).addTo(map);
  }
}
```

## Best Practices

### 1. Data Caching Strategy

```typescript
class RadarDataCache {
  private cache = new Map<string, { data: any; timestamp: number }>();
  private maxAge = 5 * 60 * 1000; // 5 minutes
  
  set(key: string, data: any) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }
  
  get(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    
    if (Date.now() - entry.timestamp > this.maxAge) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.data;
  }
}
```

### 2. Error Handling

```typescript
async function fetchRadarDataWithRetry(
  url: string,
  maxRetries: number = 3
): Promise<any> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
```

### 3. Performance Optimization

```typescript
// Use Web Workers for heavy processing
// worker.js
self.onmessage = function(e) {
  const { radarData, colorMap } = e.data;
  const processedData = processRadarData(radarData, colorMap);
  self.postMessage(processedData);
};

// main.js
const worker = new Worker('worker.js');
worker.postMessage({ radarData, colorMap: 'precipitation' });
worker.onmessage = (e) => {
  renderProcessedData(e.data);
};
```

## Important Notes

1. **Coordinate System**: All coordinates are in WGS84 (standard lat/lon)
2. **Data Resolution**: Radar data is 64x64 pixels covering ~150km radius
3. **Update Frequency**: Real radar updates every 5-10 minutes
4. **Intensity Values**: Normalized 0-1 range, representing precipitation intensity
5. **Site Coverage**: Each site covers approximately 300km diameter circle

## Example Full Integration

```typescript
import React, { useEffect, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const WeatherRadarMap: React.FC = () => {
  const [map, setMap] = useState<L.Map | null>(null);
  const [radarData, setRadarData] = useState<RawRadarDataResponse | null>(null);
  const [predictions, setPredictions] = useState<NowcastingPredictionResponse | null>(null);
  
  useEffect(() => {
    // Initialize map
    const mapInstance = L.map('map').setView([26.5, -80.5], 7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(mapInstance);
    setMap(mapInstance);
    
    // Fetch and display radar data
    fetchAndDisplayRadar(mapInstance);
    
    // Set up auto-refresh
    const interval = setInterval(() => {
      fetchAndDisplayRadar(mapInstance);
    }, 5 * 60 * 1000); // Every 5 minutes
    
    return () => {
      clearInterval(interval);
      mapInstance.remove();
    };
  }, []);
  
  const fetchAndDisplayRadar = async (mapInstance: L.Map) => {
    try {
      // Get current radar data
      const radarResponse = await fetch(
        `${NOWCASTING_BASE}/radar-data/KAMX?max_frames=1`
      );
      const radarData = await radarResponse.json();
      setRadarData(radarData);
      
      // Get predictions
      const predictionResponse = await fetch(`${NOWCASTING_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ site_id: 'KAMX', use_latest_data: true })
      });
      const predictionData = await predictionResponse.json();
      setPredictions(predictionData);
      
      // Render current radar
      if (radarData.success && radarData.frames.length > 0) {
        renderRadarFrame(mapInstance, radarData.frames[0], 'current');
      }
      
      // Render predictions
      if (predictionData.success) {
        predictionData.prediction_frames.forEach((frame, idx) => {
          renderPredictionFrame(mapInstance, frame, idx);
        });
      }
    } catch (error) {
      console.error('Failed to fetch radar data:', error);
    }
  };
  
  const renderRadarFrame = (
    mapInstance: L.Map,
    frame: RadarDataFrame,
    className: string
  ) => {
    // Clear existing layers
    mapInstance.eachLayer(layer => {
      if (layer.options?.className === className) {
        mapInstance.removeLayer(layer);
      }
    });
    
    // Create and add new overlay
    const canvas = renderRadarToCanvas(frame.data, 'precipitation');
    const bounds = L.latLngBounds(
      [frame.coordinates.bounds[2], frame.coordinates.bounds[0]],
      [frame.coordinates.bounds[3], frame.coordinates.bounds[1]]
    );
    
    L.imageOverlay(canvas.toDataURL(), bounds, {
      opacity: 0.7,
      className
    }).addTo(mapInstance);
  };
  
  const renderPredictionFrame = (
    mapInstance: L.Map,
    frameData: number[][][][],
    index: number
  ) => {
    // Convert prediction data to 2D array (remove batch and channel dimensions)
    const data2D = frameData[0].map(row => row.map(col => col[0]));
    
    // Use site coordinates from the current radar data
    if (radarData?.frames[0]) {
      const mockFrame: RadarDataFrame = {
        timestamp: new Date().toISOString(),
        data: data2D,
        coordinates: radarData.frames[0].coordinates,
        intensity_range: [0, 1],
        data_quality: 'good'
      };
      
      renderRadarFrame(mapInstance, mockFrame, `prediction-${index}`);
    }
  };
  
  return (
    <div>
      <div id="map" style={{ height: '600px', width: '100%' }} />
      <div className="controls">
        <button onClick={() => map && fetchAndDisplayRadar(map)}>
          Refresh Data
        </button>
      </div>
    </div>
  );
};

export default WeatherRadarMap;
```

## Summary

The SkyGuard Analytics Nowcasting APIs provide comprehensive weather radar data as raw arrays, enabling you to build sophisticated, interactive weather visualizations entirely in the frontend. Focus on using the raw data endpoints (#13-16) rather than the server-rendered visualization endpoints (#10-11). This approach gives you complete control over the rendering, animation, and user interaction with the weather data.

Key endpoints to prioritize:
1. `/radar-data/{site_id}` - Primary data source
2. `/radar-data/multi-site` - For mosaics
3. `/predict` - For ML predictions
4. `/radar-timeseries/{site_id}` - For animations

Always render data client-side using Canvas, WebGL, or SVG overlays on your mapping library of choice.