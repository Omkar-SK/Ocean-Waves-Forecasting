"""
OceanCast AI - FastAPI Application Entry Point.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import ALLOWED_ORIGINS
from backend.routers import forecast, health, historical, metrics
from backend.services.model_service import model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("OceanCastAI.Backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models and scalers on startup."""
    logger.info("Initializing OceanCast AI backend services...")
    try:
        scaler = model_service.get_scaler()
        logger.info("Fitted TimeSeriesScaler loaded successfully: %d features", len(scaler.feature_names))
    except Exception as e:
        logger.warning("Scaler pre-warm warning: %s", e)

    yield
    logger.info("Shutting down OceanCast AI backend.")


app = FastAPI(
    title="OceanCast AI - Deep Learning Ocean Wave Forecasting API",
    description=(
        "Production-grade REST API serving multi-horizon ocean wave forecasts "
        "using trained ANN, LSTM, and 1D-CNN architectures benchmarked on ERA5 reanalysis."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for React + Vite development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler to avoid exposing filesystem paths in errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception at %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if "Error" in type(exc).__name__ else "An unexpected processing error occurred.",
            "path": request.url.path,
        },
    )


# Mount API routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(forecast.router)
app.include_router(historical.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "OceanCast AI - Ocean Wave Forecasting API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/api/health",
        "endpoints": {
            "metadata": "/api/metadata",
            "metrics": "/api/metrics/multi-horizon",
            "live_forecast": "/api/forecast/live",
            "historical_predictions": "/api/historical/predictions",
            "historical_summary": "/api/historical/summary",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
