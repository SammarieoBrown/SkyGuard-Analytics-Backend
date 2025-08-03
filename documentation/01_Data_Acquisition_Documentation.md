# NEXRAD Data Acquisition Documentation

## Project Overview
This document details the comprehensive data acquisition process for a deep learning weather forecasting system using ConvLSTM (Convolutional Long Short-Term Memory) networks for short-term precipitation nowcasting.

## Data Source

### Primary Source: Google Cloud Platform (GCP) Public Dataset
- **Source URL**: `https://storage.googleapis.com/gcp-public-data-nexrad-l2/`
- **Data Type**: NEXRAD Level-II Radar Data
- **Format**: Archive Level-II (.ar2v files)
- **Update Frequency**: Real-time (5-minute intervals)
- **Coverage**: Continental United States
- **Temporal Range**: 1991 - Present

### Why GCP Over NOAA Direct Access?
1. **Performance**: GCP CDN provides significantly faster download speeds
2. **Reliability**: Higher uptime and more stable connections
3. **Organization**: Better structured file hierarchy (YYYY/MM/DD/SITE/)
4. **Efficiency**: Hourly tar bundles reduce individual file requests
5. **Cost**: Free public dataset access

## Radar Site Selection

### Selected Sites (5 Total)
| Site ID | Location | Name | Coordinates | Justification |
|---------|----------|------|-------------|---------------|
| KATX | Seattle, WA | Pacific Northwest | 47.68°N, 122.50°W | Complex terrain effects, mountain-influenced weather |
| PHWA | Kohala, HI | Hawaii | 19.95°N, 155.57°W | Tropical/subtropical patterns, island convection |
| KAMX | Miami, FL | Atlantic Coast | 25.61°N, 80.41°W | Hurricane activity, tropical storm development |
| KTBW | Tampa, FL | Gulf Coast | 27.71°N, 82.40°W | Gulf of Mexico storms, sea breeze convection |
| KJGX | Robins AFB, GA | Southeast Interior | 32.68°N, 83.35°W | Severe thunderstorms, tornado activity |

### Site Selection Criteria
1. **Geographic Diversity**: Coverage across major climate zones
2. **Weather Pattern Variety**: Different storm types and formation mechanisms
3. **Seasonal Variability**: Complementary seasonal patterns
4. **Storm Frequency**: High-activity areas for rich training data
5. **Terrain Influence**: Mix of coastal, inland, and complex terrain

### Coverage Analysis
- **Latitudinal Range**: 19.95°N to 47.68°N (27.73° span)
- **Longitudinal Range**: 155.57°W to 80.41°W (75.16° span)
- **Climate Zones**: Tropical, Subtropical, Temperate Maritime, Continental
- **Storm Types**: Hurricanes, Thunderstorms, Frontal systems, Orographic precipitation

## Temporal Coverage

### Date Range
- **Start Date**: June 11, 2024
- **End Date**: July 10, 2024
- **Duration**: 30 days
- **Total Site-Days**: 150 (5 sites × 30 days)

### Rationale for Date Selection
1. **Summer Season**: Peak convective activity period
2. **Hurricane Season**: Atlantic basin active period
3. **Consistent Coverage**: Avoid winter data gaps in northern sites
4. **Temporal Continuity**: Uninterrupted 30-day sequence per site
5. **Recent Data**: 2024 data ensures current radar configurations

### Temporal Resolution
- **Radar Scan Frequency**: ~4-6 minutes (varies by site/conditions)
- **Expected Files per Day**: ~200-300 per site
- **Total Expected Files**: ~30,000-45,000 across all sites
- **Time Coverage**: 24/7 continuous operation

## Data Characteristics

### NEXRAD Level-II Specifications
- **Format**: Archive Level-II (WSR-88D)
- **File Extension**: .ar2v (Archive Radar Version)
- **Compression**: None (raw binary format)
- **File Size**: 3-15MB per file (varies by precipitation activity)

### Spatial Characteristics
- **Coordinate System**: Polar (range/azimuth from radar)
- **Maximum Range**: ~460 km (varies by product)
- **Azimuth Resolution**: 0.5° (720 rays per sweep)
- **Range Resolution**: 250m (reflectivity), 125m (velocity)
- **Elevation Angles**: Multiple sweeps (typically 9-14 elevations)

### Data Products Available
| Product | Description | Units | Typical Range | Use Case |
|---------|-------------|-------|---------------|----------|
| REF | Base Reflectivity | dBZ | -30 to +70 | Precipitation intensity |
| VEL | Base Velocity | m/s | -64 to +64 | Wind patterns, storm motion |
| SW | Spectrum Width | m/s | 0 to 30 | Turbulence, data quality |
| ZDR | Differential Reflectivity | dB | -5 to +8 | Particle size/shape |
| PHI | Differential Phase | degrees | 0 to 360 | Precipitation type |
| RHO | Correlation Coefficient | unitless | 0 to 1 | Data quality indicator |

## Data Processing Pipeline

### Processing Approach
Our pipeline converts raw NEXRAD data into training-ready image sequences for ConvLSTM models through several transformation steps.

### Step 1: Data Download
```
GCP Storage → Local Storage
- Hourly tar files downloaded per site
- Extracted to organized directory structure
- Structure: downloads/SITE/YYYY-MM-DD/individual_files
```

### Step 2: Spatial Processing
```
Raw Polar Data → Cartesian Grid → Cropped Image → Resized Array
460km radius → 300km×300km → 150km×150km → 100×100 pixels
```

#### Spatial Downsampling Rationale
1. **Computational Efficiency**: 100×100 manageable for GPU memory
2. **Essential Pattern Preservation**: Maintains storm structure and motion
3. **Consistent Scale**: Uniform resolution across all sites
4. **ConvLSTM Optimization**: Standard size for temporal convolutions
5. **Regional Focus**: Captures local weather (150km radius ≈ 2-hour forecast range)

### Step 3: Data Product Selection
**Selected**: REF (Base Reflectivity) only

#### Single Channel Rationale
1. **Primary Signal**: Reflectivity directly indicates precipitation
2. **Model Simplicity**: Reduces complexity for initial implementation
3. **Training Efficiency**: Faster convergence with fewer parameters
4. **Interpretability**: Clear relationship between input and forecast target
5. **Baseline Establishment**: Foundation for future multi-channel expansion

#### Future Multi-Channel Considerations
- VEL: Storm motion and wind patterns
- ZDR: Precipitation type classification
- SW: Storm intensity and complexity indicators

### Step 4: Value Transformation
```
Raw dBZ Values → Matplotlib Visualization → RGB Image → Grayscale → Normalized Array
Float64 (-30 to +70) → RGBA (0-255) → RGB → Uint8 → Float32 (0-1)
```

#### Transformation Rationale
1. **Visualization-Based**: Leverages proven meteorological color scales
2. **Non-linear Mapping**: Better represents human perception of intensity
3. **Noise Reduction**: Matplotlib rendering smooths measurement artifacts
4. **Standardization**: Consistent appearance across different radar sites
5. **Deep Learning Compatibility**: Standard image format for CNNs

## Output Format: NumPy Arrays (.npy)

### File Format Specifications
- **Extension**: .npy (NumPy binary format)
- **Data Type**: Float32
- **Shape**: (n_samples, 100, 100)
- **Value Range**: 0.0 to 1.0 (normalized)
- **Endianness**: Native system byte order

### NumPy Format Advantages
1. **Efficiency**: Fast binary read/write operations
2. **Memory Layout**: C-contiguous arrays for optimal access
3. **Compression**: Built-in array compression support
4. **Cross-Platform**: Consistent format across operating systems
5. **Deep Learning Integration**: Direct compatibility with TensorFlow/PyTorch

### File Organization Structure
```
processed_data/
├── KATX_data_20240611_30days.npy    (~3,000 images, ~60MB)
├── PHWA_data_20240611_30days.npy    (~5,300 images, ~55MB)
├── KAMX_data_20240611_30days.npy    (~6,500 images, ~65MB)
├── KTBW_data_20240611_30days.npy    (~2,500 images, ~55MB)
└── KJGX_data_20240611_30days.npy    (~2,900 images, ~60MB)
```

## Data Volume Analysis

### Raw Data Characteristics
- **Total Raw Size**: ~150GB (estimated completion)
- **Files per Site**: 5,000-7,000 files
- **Average File Size**: 3-15MB
- **Storage Format**: Binary (.ar2v)

### Processed Data Characteristics
- **Total Processed Size**: ~300MB
- **Compression Ratio**: 500:1
- **Memory per Image**: 10KB (100×100×1 byte)
- **Total Training Samples**: ~20,000-30,000 images

### Sample Size by Site
| Site | Expected Files | Processed Size | Training Samples |
|------|---------------|----------------|------------------|
| KATX | ~3,000 | 60MB | ~3,000 |
| PHWA | ~5,300 | 55MB | ~5,300 |
| KAMX | ~6,500 | 65MB | ~6,500 |
| KTBW | ~2,500 | 55MB | ~2,500 |
| KJGX | ~2,900 | 60MB | ~2,900 |
| **Total** | **~20,200** | **~295MB** | **~20,200** |

## Quality Assurance

### Data Validation Steps
1. **File Integrity**: Verify .ar2v format headers
2. **Temporal Continuity**: Check for missing time periods
3. **Spatial Consistency**: Validate coordinate transformations
4. **Value Ranges**: Ensure realistic dBZ values
5. **Processing Success**: Confirm image generation completion

### Known Limitations
1. **Weather Dependence**: Fewer samples during clear weather
2. **Seasonal Bias**: Summer-focused data collection
3. **Geographic Bias**: US-centric radar network
4. **Temporal Resolution**: ~5-minute intervals (not continuous)
5. **Range Limitations**: Effective range varies by atmospheric conditions

## Technical Implementation

### Download Scripts
- `download_gcp_organized.py`: GCP-based parallel downloader
- `analyze_nexrad_structure.py`: Data structure analysis tool

### Processing Pipeline
- Individual site processing with spatial transformations
- Batch processing for entire datasets
- Quality control and validation steps

### Dependencies
- MetPy: NEXRAD file reading and processing
- Matplotlib: Visualization and color mapping
- OpenCV: Image processing and transformations
- NumPy: Array operations and file I/O
- Requests: HTTP downloads from GCP

## Future Enhancements

### Data Expansion Opportunities
1. **Extended Temporal Range**: Additional months/seasons
2. **Multi-Channel Processing**: Include VEL, ZDR, SW products
3. **Higher Resolution**: 200×200 or 256×256 images
4. **Additional Sites**: Expand geographic coverage
5. **Real-Time Integration**: Live data streaming capabilities

### Processing Improvements
1. **Advanced Filtering**: Remove anomalous propagation
2. **Quality Control**: Automated data quality assessment
3. **Meteorological Preprocessing**: Apply standard radar corrections
4. **Ensemble Methods**: Multiple processing approaches
5. **Data Augmentation**: Synthetic sample generation

---

**Document Version**: 1.0  
**Last Updated**: August 2024  
**Authors**: SkyGuard Analytics Team  
**Review Status**: Draft