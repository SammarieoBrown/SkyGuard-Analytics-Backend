from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import logging
import json

from app.api.v1.router import api_router
from app.config import API_V1_STR
from app.utils import NumpyEncoder
from app.core.models.model_manager import model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON response class that uses our NumpyEncoder
class NumpyJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=NumpyEncoder,
        ).encode("utf-8")

# Create FastAPI app
app = FastAPI(
    title="SkyGuard Analytics API",
    description="""
    API for the SkyGuard Analytics platform - a comprehensive severe weather prediction and analysis system.
    
    ## Features
    
    * **Impact Prediction**: Property damage and casualty risk assessment
    * **Risk Analysis**: Regional weather risk scoring and analysis  
    * **Weather Nowcasting**: Real-time precipitation forecasting using NEXRAD radar data
    * **Scenario Simulation**: Weather scenario modeling and impact analysis
    
    ## Weather Nowcasting
    
    The nowcasting system provides 6-frame precipitation predictions using:
    - **NEXRAD Level-II radar data** from KAMX (Miami) and KATX (Seattle)
    - **MinimalConvLSTM model** trained on real weather data with 99.98% accuracy
    - **Real-time processing** with automatic data pipeline from GCP public storage
    
    ## Endpoints
    
    - `/api/v1/nowcasting/predict` - Generate weather predictions
    - `/api/v1/nowcasting/sites` - Get supported radar sites
    - `/api/v1/nowcasting/health` - Check system health
    """,
    version="1.0.0",
    default_response_class=NumpyJSONResponse,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=API_V1_STR)


@app.on_event("startup")
async def startup_event():
    """
    Preload all models at startup to avoid timeout issues on first request.
    This ensures models are loaded into memory before serving any requests.
    """
    logger.info("Starting application - preloading models...")
    
    try:
        # Preload property damage model
        logger.info("Loading property damage model...")
        property_damage_model = model_manager.get_property_damage_model()
        logger.info("✅ Property damage model loaded successfully")
        
        # Preload casualty risk model
        logger.info("Loading casualty risk model...")
        casualty_risk_model = model_manager.get_casualty_risk_model()
        logger.info("✅ Casualty risk model loaded successfully")
        
        # Preload severity model
        logger.info("Loading severity model...")
        severity_model = model_manager.get_severity_model()
        logger.info("✅ Severity model loaded successfully")
        
        # Preload weather nowcasting model
        logger.info("Loading weather nowcasting model...")
        nowcasting_model = model_manager.get_weather_nowcasting_model()
        logger.info("✅ Weather nowcasting model loaded successfully")
        
        # Get model status
        model_status = model_manager.get_model_health_status()
        logger.info(f"All models loaded. Status: {model_status}")
        
    except Exception as e:
        logger.error(f"Failed to preload models at startup: {str(e)}")
        logger.warning("Application starting with models loading on-demand. First requests may be slow.")
    
    logger.info("Application startup complete")


@app.get("/")
async def root()
    """Root endpoint with API information."""
    return {
        "name": "SkyGuard Analytics API",
        "version": "1.0.0",
        "status": "active",
        "docs_url": "/docs",
        "description": "API for severe weather prediction and analysis with real-time nowcasting",
        "features": [
            "Impact Prediction",
            "Risk Analysis", 
            "Weather Nowcasting",
            "Scenario Simulation"
        ],
        "nowcasting_sites": ["KAMX (Miami)", "KATX (Seattle)"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
