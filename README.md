# SkyGuard Analytics

An integrated severe weather prediction and analysis platform.

## Overview

SkyGuard Analytics provides API endpoints for:
- Property damage prediction
- Casualty risk assessment
- Weather severity classification
- Regional risk scoring
- Scenario simulation

## Setup and Installation

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the API**

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

- Interactive API documentation (Swagger UI): `http://localhost:8000/docs`
- Alternative documentation (ReDoc): `http://localhost:8000/redoc`

## Endpoints

- Property Damage Prediction: `POST /api/v1/impact/property-damage`
- More endpoints coming soon...

## Usage Examples

### Predict Property Damage

```bash
curl -X POST "http://localhost:8000/api/v1/impact/property-damage" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "Thunderstorm Wind",
    "state": "TX",
    "magnitude": 65.0,
    "duration_hours": 1.5,
    "month": 6,
    "hour": 14,
    "latitude": 32.7767,
    "longitude": -96.7970
  }'
``` 