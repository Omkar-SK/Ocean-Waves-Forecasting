"""
Historical predictions, scatter plots, residual distributions, and export endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.config import SUPPORTED_HORIZONS
from backend.schemas.historical import (
    HistoricalPredictionsResponse,
    HistoricalSummaryResponse,
    ResidualHistogramBin,
    ScatterPoint,
)
from backend.services.historical_service import historical_service

router = APIRouter(prefix="/api/historical", tags=["Historical Evaluation"])


@router.get("/summary", response_model=HistoricalSummaryResponse)
def get_historical_summary(
    horizon: int = Query(1, description="Horizon in hours: 1, 6, or 24")
):
    """Returns summary statistics for the historical test split predictions."""
    try:
        return historical_service.get_summary(horizon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load historical summary: {str(e)}")


@router.get("/predictions", response_model=HistoricalPredictionsResponse)
def get_historical_predictions(
    horizon: int = Query(1, description="Horizon: 1, 6, or 24"),
    model: Optional[str] = Query(None, description="Model: 'lstm', 'ann', 'cnn1d', 'all'"),
    start_date: Optional[str] = Query(None, description="Optional start ISO date (e.g. 2024-08-01)"),
    end_date: Optional[str] = Query(None, description="Optional end ISO date (e.g. 2024-08-31)"),
    downsample: Optional[int] = Query(None, description="Max points to return for charting performance"),
    max_points: Optional[int] = Query(None, description="Alias for downsample"),
):
    """Returns historical actual vs predicted time series points."""
    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(status_code=400, detail=f"Invalid horizon {horizon}h.")
    try:
        eff_downsample = max_points or downsample or 500
        return historical_service.get_time_series(
            horizon=horizon,
            start_date=start_date,
            end_date=end_date,
            downsample=eff_downsample,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query historical predictions: {str(e)}")


@router.get("/scatter", response_model=List[ScatterPoint])
def get_scatter_data(
    horizon: int = Query(1, description="Horizon: 1, 6, or 24"),
    model: str = Query("lstm", description="Model: 'lstm', 'ann', or 'cnn1d'"),
    sample_size: Optional[int] = Query(None, description="Evenly sampled pairs count"),
    max_points: Optional[int] = Query(None, description="Alias for sample_size"),
):
    """Returns sampled actual vs predicted pairs for correlation scatter plotting."""
    try:
        eff_size = max_points or sample_size or 800
        clean_model = model.lower().replace("wave", "").strip()
        if clean_model not in ["lstm", "ann", "cnn1d"]:
            clean_model = "lstm"
        return historical_service.get_scatter(horizon=horizon, model=clean_model, sample_size=eff_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate scatter data: {str(e)}")


@router.get("/residuals", response_model=List[ResidualHistogramBin])
def get_residual_histogram(
    horizon: int = Query(1, description="Horizon: 1, 6, or 24"),
    model: str = Query("lstm", description="Model: 'lstm', 'ann', or 'cnn1d'"),
    num_bins: Optional[int] = Query(None, description="Number of histogram bins"),
    bins: Optional[int] = Query(None, description="Alias for num_bins"),
):
    """Returns error residual distribution histogram bins."""
    try:
        eff_bins = bins or num_bins or 30
        clean_model = model.lower().replace("wave", "").strip()
        if clean_model not in ["lstm", "ann", "cnn1d"]:
            clean_model = "lstm"
        return historical_service.get_residuals(horizon=horizon, model=clean_model, num_bins=eff_bins)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute residual bins: {str(e)}")


@router.get("/export")
def export_predictions_csv(
    horizon: int = Query(1, description="Horizon: 1, 6, or 24")
):
    """Downloads the complete un-downsampled test set predictions CSV."""
    try:
        csv_file = historical_service.get_csv_filepath(horizon)
        filename = f"era5_wave_predictions_h{horizon}.csv"
        return FileResponse(
            path=str(csv_file),
            filename=filename,
            media_type="text/csv",
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export CSV: {str(e)}")
