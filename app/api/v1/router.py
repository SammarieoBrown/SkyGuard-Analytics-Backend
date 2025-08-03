from fastapi import APIRouter
from app.api.v1.endpoints import impact, risk, simulation, nowcasting

api_router = APIRouter()

# Include router for impact prediction endpoints
api_router.include_router(
    impact.router,
    prefix="/impact",
    tags=["impact"]
)

# Include router for risk assessment endpoints
api_router.include_router(
    risk.router,
    prefix="/risk",
    tags=["risk"]
)

# Include router for scenario simulation endpoints
api_router.include_router(
    simulation.router,
    prefix="/simulation",
    tags=["simulation"]
)

# Include router for weather nowcasting endpoints
api_router.include_router(
    nowcasting.router,
    prefix="/nowcasting",
    tags=["nowcasting"]
)

# Include additional routers as they are developed:
# api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"]) 