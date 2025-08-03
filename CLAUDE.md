# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start the API server:**
```bash
uvicorn main:app --reload
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run tests:**
```bash
python test_all_endpoints_complete.py
```

**For Deep Learning models (NEXRAD data processing):**
```bash
cd "Deep-Learning-for-Weather-and-Climate-Science"
pip install -r requirements.txt
```

## Architecture Overview

### Main API Application (`/app`)
- **FastAPI-based** severe weather prediction and analysis platform
- **Modular router structure** with three main API domains:
  - `/api/v1/impact` - Property damage, casualty risk, severity predictions
  - `/api/v1/risk` - Regional risk scoring and state-level assessments  
  - `/api/v1/simulation` - Scenario modeling and sensitivity analysis
- **Singleton ModelManager** (`app/core/models/model_manager.py`) prevents duplicate model loading
- **Custom NumpyJSONResponse** for handling NumPy arrays in API responses

### Machine Learning Models (`/app/models`)
- **Pre-trained sklearn models** stored as pickle files:
  - Property damage: HistGradientBoosting model
  - Casualty risk: Custom risk assessment model
  - Severity classification: Best performing severity model
  - State risk scores: Regional risk scoring data

### Deep Learning Research (`/Deep-Learning-for-Weather-and-Climate-Science`)
- **Convolutional LSTM Encoder-Decoder** for weather nowcasting
- **NEXRAD radar data processing** pipeline for real-time weather prediction
- **Multi-site radar data** (KATX-Seattle, PHWA-Hawaii, KAMX-Miami)
- **Spatiotemporal sequence forecasting** for precipitation prediction

### Configuration & Data
- **Environment-based config** (`app/config.py`) supports SQLite (dev) and PostgreSQL (prod)
- **Model paths configured centrally** with automatic path resolution
- **CORS enabled** for cross-origin requests (set to "*" - should be restricted in production)

### Key Integration Points
- **Custom preprocessing pipeline** (`app/core/preprocessing.py`) for input normalization
- **Service layer pattern** separating business logic from API endpoints
- **Database models** for prediction history and scenario storage
- **Comprehensive test coverage** with endpoint validation script

The codebase combines operational ML models for immediate weather impact assessment with research-grade deep learning models for advanced weather prediction capabilities.