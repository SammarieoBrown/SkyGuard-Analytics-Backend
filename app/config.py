from pathlib import Path
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent

# Database Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "development":
    # Use SQLite for local development
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR}/skyguard_dev.db"
    )
else:
    # Use PostgreSQL for production
    PGHOST = os.getenv("PGHOST", "localhost")
    PGDATABASE = os.getenv("PGDATABASE", "skyguardanalytics")
    PGUSER = os.getenv("PGUSER", "postgres")
    PGPASSWORD = os.getenv("PGPASSWORD", "")
    PGPORT = int(os.getenv("PGPORT", "5432"))
    PGSSLMODE = os.getenv("PGSSLMODE", "prefer")
    
    # Construct PostgreSQL DATABASE_URL
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}?sslmode={PGSSLMODE}"
    )

# Application settings
APP_NAME = os.getenv("APP_NAME", "SkyGuard Analytics")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model paths (using local paths within SkyGuard Analytics)
MODELS_DIR = BASE_DIR / "app" / "models"
PROPERTY_DAMAGE_MODEL_PATH = MODELS_DIR / "property_damage" / "histgradientboosting_model.pkl"
CASUALTY_RISK_MODEL_PATH = MODELS_DIR / "casualty_risk" / "casualty_risk_model.pkl"
SEVERITY_MODEL_PATH = MODELS_DIR / "severity" / "best_severity_model.pkl"
STATE_RISK_SCORES_PATH = MODELS_DIR / "risk_scoring" / "state_risk_scores.pkl"

# Weather Nowcasting Model Configuration
WEATHER_NOWCASTING_MODEL_DIR = BASE_DIR / "models" / "weather_nowcasting"
WEATHER_NOWCASTING_MODEL_PATH = WEATHER_NOWCASTING_MODEL_DIR / "best_model.keras"
WEATHER_NOWCASTING_CONFIG_PATH = WEATHER_NOWCASTING_MODEL_DIR / "model_config.json"

# NEXRAD Data Configuration
NEXRAD_DATA_DIR = BASE_DIR / "app" / "data" / "radar"
NEXRAD_GCP_BASE_URL = "https://storage.googleapis.com/gcp-public-data-nexrad-l2"
NEXRAD_SUPPORTED_SITES = ["KAMX", "KATX"]  # Miami and Seattle
NEXRAD_DATA_RETENTION_DAYS = int(os.getenv("NEXRAD_DATA_RETENTION_DAYS", "7"))
NEXRAD_MAX_DOWNLOAD_WORKERS = int(os.getenv("NEXRAD_MAX_DOWNLOAD_WORKERS", "4"))

# Radar Processing Configuration
RADAR_OUTPUT_SIZE = (64, 64)  # Model input size
RADAR_RANGE_LIMIT_KM = 150
RADAR_SEQUENCE_LENGTH = 10  # Input sequence length for model
RADAR_PREDICTION_LENGTH = 6  # Output prediction length

# API settings
API_V1_STR = os.getenv("API_V1_STR", "/api/v1")

# GCS Configuration
GCS_BUCKET_NAME = "skyguard-capstone"  # Hardcoded bucket name
GCS_CREDENTIALS = os.getenv("GCP_STORAGE")  # JSON string with service account credentials
USE_GCS_STORAGE = True  # Hardcoded to always use GCS
GCS_DATA_RETENTION_DAYS = 7  # Hardcoded retention period 