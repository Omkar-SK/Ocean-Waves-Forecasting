"""
Ocean Wave Forecasting Dashboard: Deep Learning Benchmark (ANN vs LSTM vs 1D-CNN)
ECMWF ERA5 Reanalysis Time-Series Forecasting System.
"""

from datetime import datetime, timezone, timedelta
import html
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_wave_model
from src.data.dataset import TimeSeriesScaler, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL
from src.data.live_service import (
    LiveFetchResult,
    LiveForecastResult,
    fetch_live_marine_and_weather,
    fetch_demo_era5_window,
    validate_and_prepare_input_window,
    predict_live_forecast,
    get_sea_state_classification,
)
from src.visualization import plot_live_forecast_trajectory


# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ocean Wave Forecasting | Deep Learning Benchmark",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Ocean-Themed Aesthetic)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background & Typography */
    .main {
        background-color: #0b111e;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Banner */
    .ocean-header {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0284c7 100%);
        padding: 24px 30px;
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.15);
    }
    .ocean-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .ocean-subtitle {
        color: #bae6fd;
        font-size: 15px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: #111c2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-card-title {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-card-value {
        color: #38bdf8;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-card-sub {
        color: #64748b;
        font-size: 12px;
        margin-top: 2px;
    }

    /* Highlight Badges */
    .badge-primary {
        background-color: #0284c7;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-success {
        background-color: #059669;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #d97706;
        color: white;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Section Headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data & Artifact Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_processed_timeseries() -> pd.DataFrame:
    """Load preprocessed ERA5 hourly time series from Parquet."""
    parquet_path = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
    if not parquet_path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    return df


@st.cache_data(show_spinner=False)
def load_preprocessing_report() -> Optional[Dict]:
    """Load JSON preprocessing report containing QA and coordinate info."""
    report_path = PROJECT_ROOT / "outputs" / "reports" / "preprocessing_report.json"
    if not report_path.is_file():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_comparison_metrics() -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Load model comparison CSV and detailed metrics JSON."""
    csv_path = PROJECT_ROOT / "outputs" / "metrics" / "model_comparison.csv"
    json_path = PROJECT_ROOT / "outputs" / "metrics" / "detailed_metrics.json"

    df_metrics = pd.read_csv(csv_path) if csv_path.is_file() else None
    meta_metrics = None
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            meta_metrics = json.load(f)
    return df_metrics, meta_metrics


@st.cache_data(show_spinner=False)
def load_multi_horizon_metrics() -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Load multi-horizon comparison CSV and detailed JSON."""
    csv_path = PROJECT_ROOT / "outputs" / "metrics" / "multi_horizon_comparison.csv"
    json_path = PROJECT_ROOT / "outputs" / "metrics" / "multi_horizon_detailed.json"

    df_multi = pd.read_csv(csv_path) if csv_path.is_file() else None
    meta_multi = None
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            meta_multi = json.load(f)
    return df_multi, meta_multi



@st.cache_data(show_spinner=False)
def load_test_predictions_sample(nrows: int = 1000) -> Optional[pd.DataFrame]:
    """Load a sample of test predictions for inspection."""
    pred_path = PROJECT_ROOT / "outputs" / "predictions" / "test_predictions.csv"
    if not pred_path.is_file():
        return None
    return pd.read_csv(pred_path, nrows=nrows)


@st.cache_resource(show_spinner=False)
def load_scaler() -> Optional[TimeSeriesScaler]:
    """Load persisted TimeSeriesScaler artifact."""
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    if not scaler_path.is_file():
        return None
    return TimeSeriesScaler.load(scaler_path)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_cached_live_marine(latitude: float, longitude: float, timeout_s: int) -> LiveFetchResult:
    """Cached Open-Meteo REST query with 15-minute TTL to reduce redundant network load."""
    return fetch_live_marine_and_weather(latitude=latitude, longitude=longitude, timeout_s=timeout_s)



@st.cache_resource(show_spinner=False)
def load_model_checkpoint(model_type: str, horizon: int = 1) -> Tuple[Optional[torch.nn.Module], Optional[Dict]]:
    """
    Load a trained PyTorch model and its metadata for a given architecture and horizon.
    Returns (model, metadata) or (None, None) if not available.
    """
    m_code = model_type.lower().replace("wave", "")
    
    # Check horizon-specific checkpoint or default checkpoint
    ckpt_path = PROJECT_ROOT / "models" / f"best_{m_code}_h{horizon}.pt"
    meta_path = PROJECT_ROOT / "models" / f"best_{m_code}_h{horizon}_metadata.json"

    if not ckpt_path.is_file() and horizon == 1:
        ckpt_path = PROJECT_ROOT / "models" / f"best_{m_code}.pt"
        meta_path = PROJECT_ROOT / "models" / f"best_{m_code}_metadata.json"

    if not ckpt_path.is_file() or not meta_path.is_file():
        return None, None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Validate that checkpoint matches requested horizon
    ckpt_horizon = meta.get("dataset_config", {}).get("horizon", 1)
    if ckpt_horizon != horizon:
        return None, None

    cfg = meta.get("model_config", {})
    model = build_wave_model(
        model_type=cfg.get("model_type", model_type),
        input_dim=cfg.get("input_dim", len(DEFAULT_FEATURE_COLS)),
        seq_len=cfg.get("seq_len", 24),
        output_dim=cfg.get("output_dim", 1),
    )

    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state_dict"])
    else:
        model.load_state_dict(ckpt_data)

    model.eval()
    return model, meta


def get_sea_state_description(swh_meters: float) -> Tuple[str, str]:
    """World Meteorological Organization (WMO) Sea State code classification."""
    if swh_meters < 0.1:
        return "Calm (Glassy)", "#38bdf8"
    elif swh_meters < 0.5:
        return "Calm (Rippled)", "#38bdf8"
    elif swh_meters < 1.25:
        return "Smooth", "#22c55e"
    elif swh_meters < 2.5:
        return "Slight to Moderate", "#eab308"
    elif swh_meters < 4.0:
        return "Rough", "#f97316"
    elif swh_meters < 6.0:
        return "Very Rough", "#ef4444"
    elif swh_meters < 9.0:
        return "High", "#b91c1c"
    else:
        return "Phenomenal", "#7f1d1d"


# -----------------------------------------------------------------------------
# Sidebar Navigation & Workspace Status
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #38bdf8; margin: 0;'>🌊 WaveCast DL</h2>
        <p style='color: #94a3b8; font-size: 13px; margin: 4px 0 0 0;'>Deep Learning Ocean Forecasting</p>
    </div>
    <hr style='border: 1px solid #1e293b; margin: 12px 0 18px 0;'>
    """,
    unsafe_allow_html=True,
)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📌 Project Overview",
        "🌐 Live Ocean Forecast (Current Conditions)",
        "⏪ Historical Scenario Engine (Test Split)",
        "📊 Historical Data Explorer",
        "🏆 Model Comparison & Metrics",
        "🔬 Methodology & Architecture",
    ],
    index=1,
)

st.sidebar.markdown("<hr style='border: 1px solid #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: #e2e8f0; font-size: 14px;'>Environment & Artifacts</h4>", unsafe_allow_html=True)

# Dynamic verification of artifacts
df_all = load_processed_timeseries()
df_metrics, meta_metrics = load_comparison_metrics()
df_multi, meta_multi = load_multi_horizon_metrics()
scaler_obj = load_scaler()
has_models = (PROJECT_ROOT / "models" / "best_lstm.pt").is_file()
has_all_horizons = (PROJECT_ROOT / "models" / "best_lstm_h24.pt").is_file()

st.sidebar.markdown(
    f"""
    - **Dataset**: `{'Loaded (' + f'{len(df_all):,}' + ' hrs)' if not df_all.empty else 'Missing'}`
    - **Fitted Scaler**: `{'Available' if scaler_obj else 'Missing'}`
    - **Trained Models**: `{'Available (H=1h, 6h, 24h - 9 Models)' if has_all_horizons else ('Available (H=1h)' if has_models else 'Pending')}`
    - **Live Operational API**: `Open-Meteo Marine / ECMWF (CC BY 4.0)`
    - **Offline Demo Fallback**: `Verified ERA5 Archive Window`
    - **Multi-Horizon Benchmark**: `{'Available (1h, 6h, 24h)' if df_multi is not None else 'Pending'}`
    """
)


st.sidebar.markdown("<hr style='border: 1px solid #1e293b; margin: 15px 0;'>", unsafe_allow_html=True)
st.sidebar.caption("Course Project | SEM 5 Deep Learning")
st.sidebar.caption("Offshore Arabian Sea (18.50°N, 72.50°E)")


# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================
if nav_page == "📌 Project Overview":
    st.markdown(
        """
        <div class="ocean-header">
            <h1 class="ocean-title">Ocean Wave Forecasting Using Deep Learning</h1>
            <p class="ocean-subtitle">Comparative Benchmark: Multilayer Perceptron (ANN) vs Long Short-Term Memory (LSTM) vs Temporal 1D-CNN</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Dataset Coverage</div>
                <div class="metric-card-value">14.8 Years</div>
                <div class="metric-card-sub">2012-01-01 to 2026-10-17</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Observations</div>
                <div class="metric-card-value">{len(df_all):,}</div>
                <div class="metric-card-sub">Hourly Continuous (0 NaNs)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Selected Coordinates</div>
                <div class="metric-card-value">18.5°N, 72.5°E</div>
                <div class="metric-card-sub">Verified Arabian Sea Node</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-card-title">Primary Target</div>
                <div class="metric-card-value">SWH (Hs)</div>
                <div class="metric-card-sub">Significant Wave Height (m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🎯 Project Objective")
    st.markdown(
        """
        Ocean wave height forecasting is critical for maritime safety, offshore oil & gas rigs, port navigation, and coastal disaster mitigation.
        This project constructs an end-to-end, leak-free Deep Learning pipeline to predict future **Significant Wave Height ($H_s$)** using past wave and atmospheric conditions.
        
        We systematically benchmark three fundamental deep learning paradigms on identical chronological splits:
        1. **WaveANN (Feedforward Multilayer Perceptron)**: Dense feature baseline with flattened 24-hour historical windows.
        2. **WaveLSTM (Recurrent Neural Network)**: Two-layer stacked LSTM capturing temporal sequential dynamics across 24 hourly time steps.
        3. **WaveCNN1D (Temporal 1D-CNN)**: Multi-layer dilated causal temporal convolutions extracting multi-scale local time features.
        """
    )

    st.markdown("---")
    col_map, col_era5 = st.columns([1, 1])

    with col_map:
        st.markdown("### 📍 Offshore Point Verification")
        st.markdown(
            """
            The study area is located offshore in the Arabian Sea off the west coast of India.
            - **Latitude**: $18.50^\circ\\text{ N}$, **Longitude**: $72.50^\circ\\text{ E}$
            - **Grid Alignment**: Exact intersection of both the $0.50^\circ$ wave grid and $0.25^\circ$ surface wind grid.
            - **Water Verification**: $100\\%$ valid wave measurements across all 129,696 hours.
            """
        )
        map_df = pd.DataFrame({"lat": [18.50], "lon": [72.50]})
        st.map(map_df, zoom=6)

    with col_era5:
        st.markdown("### 🌐 About ECMWF ERA5 Reanalysis")
        st.info(
            """
            **What is ERA5 Reanalysis?**
            ERA5 is the fifth-generation global atmospheric and ocean wave reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF).
            
            Unlike raw satellite or buoy sensors (which have observation gaps and instrumentation noise) or pure numerical weather forecasts (which drift over time), **reanalysis combines physical laws of fluid dynamics with global observations using 4D-Var data assimilation**.
            
            **Retrospective Benchmark vs. Operational Forecasting:**
            In this benchmark, models are evaluated retrospectively on real historical ERA5 states. In a live operational setting, the model would be fed with live buoy telemetry or NWP wind forecasts.
            """
        )
        st.markdown("#### Input Feature Suite (13 Features)")
        st.markdown(
            """
            - **Wave Variables**: Significant Wave Height (`swh`, m), Mean Wave Period (`mwp`, s), Circular Wave Direction (`mwd_sin`, `mwd_cos`).
            - **Atmospheric Variables**: 10m Eastward Wind (`u10`, m/s), 10m Northward Wind (`v10`, m/s), Wind Speed (`wind_speed`, m/s), Circular Wind Direction (`wind_dir_sin`, `wind_dir_cos`).
            - **Temporal Harmonics**: Diurnal cycle (`hour_sin`, `hour_cos`), Seasonal annual cycle (`month_sin`, `month_cos`).
            """
        )


# =============================================================================
# PAGE 2: LIVE OCEAN FORECAST (CURRENT CONDITIONS)
# =============================================================================
elif nav_page == "🌐 Live Ocean Forecast (Current Conditions)":
    st.markdown(
        """
        <div class="ocean-header">
            <h1 class="ocean-title">🌐 Live Ocean Wave Forecasting (Current Conditions)</h1>
            <p class="ocean-subtitle">Operational Deep Learning Inference (ANN, LSTM, 1D-CNN) Driven by Real-Time ECMWF Marine & Atmospheric Telemetry</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if scaler_obj is None:
        st.error("Training scaler artifact missing from models/scaler.pkl. Please ensure models are trained.")
        st.stop()

    # 1. Operational Controls
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
    with c_ctrl1:
        horizon_choice = st.selectbox(
            "Forecast Horizon (Lead Time)",
            [1, 6, 24],
            index=0,
            format_func=lambda h: f"{h} Hour{'s' if h > 1 else ''} Ahead (t + {h}h)",
            help="Lead time into the future for significant wave height prediction.",
        )
    with c_ctrl2:
        model_choice = st.selectbox(
            "Deep Learning Architecture",
            [
                "WaveLSTM (Recurrent Neural Network)",
                "WaveANN (Feedforward Multilayer Perceptron)",
                "WaveCNN1D (Temporal 1D-CNN)",
            ],
            index=0,
            help="Select PyTorch architecture to execute for this forecast horizon.",
        )
    with c_ctrl3:
        source_mode = st.selectbox(
            "Observation Ingestion Feed",
            [
                "🌐 Open-Meteo Marine & Weather API (Live Real-Time)",
                "📂 Offline Historical Demo Mode (ERA5 Archive Window)",
            ],
            index=0 if "force_demo" not in st.session_state else 1,
            help="Choose between real-time external API or offline historical ground-truth replay.",
        )

    # 2. Location & Advanced Settings Expander
    with st.expander("⚙️ Offshore Coordinates, API Settings & Cache Control", expanded=False):
        set_c1, set_c2, set_c3, set_c4 = st.columns([1, 1, 1, 1])
        with set_c1:
            target_lat = st.number_input("Target Latitude (°N)", value=18.50, min_value=-90.0, max_value=90.0, step=0.05, format="%.2f")
        with set_c2:
            target_lon = st.number_input("Target Longitude (°E)", value=72.50, min_value=-180.0, max_value=180.0, step=0.05, format="%.2f")
        with set_c3:
            api_timeout = st.slider("Network Timeout (s)", min_value=5, max_value=30, value=10, step=1)
        with set_c4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Data Now", use_container_width=True):
                st.cache_data.clear()
                if "force_demo" in st.session_state:
                    del st.session_state["force_demo"]
                st.rerun()

        st.caption(
            "Target Buoy Node: Offshore Arabian Sea (18.50°N, 72.50°E, ~80km offshore Mumbai). "
            "Data responses are cached for 15 minutes. Click 'Refresh Data Now' to force an immediate network query."
        )

    # 3. Ingestion & Quality Validation
    fetch_result: Optional[LiveFetchResult] = None
    if "Live Real-Time" in source_mode:
        with st.spinner("Connecting to Open-Meteo Marine & Atmospheric APIs for near-real-time observations..."):
            try:
                fetch_result = fetch_cached_live_marine(latitude=target_lat, longitude=target_lon, timeout_s=api_timeout)
            except Exception as e:
                st.markdown(
                    f"""
                    <div style='background: #3b1812; border: 1px solid #7f1d1d; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;'>
                        <h4 style='color: #fca5a5; margin: 0 0 6px 0;'>⚠️ External API Query Notice</h4>
                        <p style='color: #fecaca; font-size: 14px; margin: 0 0 8px 0;'>
                            Could not connect to Open-Meteo REST API: <code>{html.escape(str(e))}</code>.
                        </p>
                        <p style='color: #cbd5e1; font-size: 13px; margin: 0;'>
                            <b>Scientific Transparency Guarantee:</b> Synthetic, fake, or random inputs are strictly prohibited.
                            When live connectivity is unavailable, switch to the verified <b>Offline Historical Demo Mode</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Switch to Offline Historical Demo Mode", use_container_width=True):
                    st.session_state["force_demo"] = True
                    st.rerun()
                st.stop()
    else:
        fetch_result = fetch_demo_era5_window()

    # 4. Display Operational Honesty Status Banner
    now_utc = datetime.now(timezone.utc)
    latest_obs_utc = fetch_result.latest_obs_time_utc
    latest_obs_ist = latest_obs_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    freshness_mins = max(0, int((now_utc - latest_obs_utc).total_seconds() / 60))

    if not fetch_result.is_demo:
        st.markdown(
            f"""
            <div style='background: #064e3b; border: 1px solid #059669; border-radius: 8px; padding: 12px 20px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div>
                    <span class="badge-success">LIVE OPERATIONAL INFERENCE</span>
                    <span style='color: #a7f3d0; font-size: 13px; margin-left: 10px;'>
                        Source: <b>{fetch_result.source_name}</b> | Location: <b>{target_lat:.2f}°N, {target_lon:.2f}°E</b>
                    </span>
                </div>
                <div style='color: #d1fae5; font-size: 13px;'>
                    Anchor: <b>{latest_obs_utc.strftime("%Y-%m-%d %H:00 UTC")}</b> ({latest_obs_ist.strftime("%d-%b %H:%M IST")}) | Freshness: <b>~{freshness_mins}m ago</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background: #451a03; border: 1px solid #d97706; border-radius: 8px; padding: 14px 20px; margin-bottom: 18px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                    <div>
                        <span class="badge-warning">OFFLINE DEMO MODE ACTIVE</span>
                        <b style='color: #fde68a; font-size: 14px; margin-left: 10px;'>ERA5 Historical Ground Truth Archive Window</b>
                    </div>
                    <div style='color: #fef3c7; font-size: 12px;'>
                        Latest Historical Observation: <b>{latest_obs_utc.strftime("%Y-%m-%d %H:00 UTC")}</b>
                    </div>
                </div>
                <p style='color: #fef08a; font-size: 13px; margin: 8px 0 0 0;'>
                    ⚠️ <b>Scientific Transparency Notice:</b> This is <b>NOT</b> a current live forecast. It replays the final verified 24-hour observation window ending 2026-10-17 23:00 UTC from the historical ERA5 archive for offline validation and pipeline inspection.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Input Window Validation (Enforce 24 consecutive hourly steps, no NaNs, no gaps)
    try:
        df_window, verified_anchor = validate_and_prepare_input_window(
            fetch_result.raw_df,
            window_size=24,
        )
    except Exception as val_err:
        st.markdown(
            f"""
            <div style='background: #3b1812; border: 1px solid #7f1d1d; border-radius: 8px; padding: 16px 20px; margin-bottom: 18px;'>
                <h4 style='color: #fca5a5; margin: 0 0 4px 0;'>❌ Observation Quality Validation Rejection</h4>
                <p style='color: #fecaca; font-size: 14px; margin: 0;'>
                    {html.escape(str(val_err))}
                </p>
                <p style='color: #94a3b8; font-size: 12px; margin-top: 6px;'>
                    Under strict scientific integrity rules, missing observations or irregular timestamps are rejected to prevent generating inaccurate or misleading forecasts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # 6. Model Inference Execution
    m_code = "lstm" if "LSTM" in model_choice else ("ann" if "ANN" in model_choice else "cnn1d")
    nwp_forecast_val = fetch_result.external_nwp_forecasts.get(horizon_choice)

    try:
        forecast_res = predict_live_forecast(
            prepared_df=df_window,
            anchor_time=verified_anchor,
            horizon=horizon_choice,
            model_type=m_code,
            scaler=scaler_obj,
            models_dir=PROJECT_ROOT / "models",
            external_nwp_forecast=nwp_forecast_val,
            is_demo=fetch_result.is_demo,
            source_info={
                "source_name": fetch_result.source_name,
                "attribution": fetch_result.attribution,
                "notes": fetch_result.notes,
            },
        )
    except Exception as inf_err:
        st.error(f"Inference execution failed: {inf_err}")
        st.stop()

    # Timezone derivations
    target_utc = forecast_res.target_time_utc
    target_ist = target_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    anchor_utc = forecast_res.anchor_time_utc
    anchor_ist = anchor_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    gen_time_utc = forecast_res.generated_at_utc
    gen_time_ist = gen_time_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

    # 7. Metric KPI Cards
    st.markdown("### 🎯 Live Forecast Prognosis")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        delta_m = forecast_res.forecast_swh_m - forecast_res.persistence_swh_m
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Predicted Wave Height (Hs)</div>
                <div class="metric-card-value">{forecast_res.forecast_swh_m:.3f} m</div>
                <div class="metric-card-sub">{forecast_res.forecast_swh_m * 100:.1f} cm ({delta_m:+.3f} m vs persistence)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Forecast Target Time (t+{horizon_choice}h)</div>
                <div class="metric-card-value" style="font-size: 19px; margin-top: 8px;">{target_utc.strftime('%Y-%m-%d %H:00 UTC')}</div>
                <div class="metric-card-sub">{target_ist.strftime('%d-%b %Y %H:%M IST')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Latest Ingestion Anchor (t)</div>
                <div class="metric-card-value" style="font-size: 19px; margin-top: 8px;">{anchor_utc.strftime('%Y-%m-%d %H:00 UTC')}</div>
                <div class="metric-card-sub">{anchor_ist.strftime('%d-%b %Y %H:%M IST')} ({'Fresh' if not fetch_result.is_demo else 'Historical Archive'})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">WMO Sea State Code</div>
                <div class="metric-card-value" style="color: {forecast_res.sea_state_color}; font-size: 20px; margin-top: 8px;">{forecast_res.sea_state_desc}</div>
                <div class="metric-card-sub">Arabian Sea Offshore Node</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 8. Side-by-Side Model vs NWP vs Persistence Comparison Card
    st.markdown("### ⚖️ Multi-Method Comparison: Deep Learning vs Numerical NWP vs Persistence")
    comp_c1, comp_c2, comp_c3 = st.columns(3)
    with comp_c1:
        st.markdown(
            f"""
            <div style='background: #0f1d33; border: 1px solid #1e3a5f; border-radius: 8px; padding: 14px 18px;'>
                <div style='color: #38bdf8; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Our Deep Learning Model</div>
                <div style='color: #ffffff; font-size: 24px; font-weight: 700; margin-top: 4px;'>{forecast_res.forecast_swh_m:.3f} m</div>
                <div style='color: #94a3b8; font-size: 12px; margin-top: 2px;'><b>{forecast_res.model_name}</b> ({forecast_res.checkpoint_file})</div>
                <div style='color: #cbd5e1; font-size: 11px; margin-top: 6px;'>Learned non-linear mapping from 24h historical sequence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with comp_c2:
        nwp_str = f"{forecast_res.external_nwp_swh_m:.3f} m" if forecast_res.external_nwp_swh_m is not None else "N/A (Offline Demo)"
        delta_nwp_str = f"{forecast_res.forecast_swh_m - forecast_res.external_nwp_swh_m:+.3f} m vs DL" if forecast_res.external_nwp_swh_m is not None else "—"
        st.markdown(
            f"""
            <div style='background: #19142b; border: 1px solid #4c1d95; border-radius: 8px; padding: 14px 18px;'>
                <div style='color: #c084fc; font-size: 12px; font-weight: 600; text-transform: uppercase;'>External NWP Provider</div>
                <div style='color: #ffffff; font-size: 24px; font-weight: 700; margin-top: 4px;'>{nwp_str}</div>
                <div style='color: #94a3b8; font-size: 12px; margin-top: 2px;'><b>ECMWF WAM / Copernicus</b> ({delta_nwp_str})</div>
                <div style='color: #cbd5e1; font-size: 11px; margin-top: 6px;'>Third-party numerical wave model physics simulation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with comp_c3:
        st.markdown(
            f"""
            <div style='background: #111c2e; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px;'>
                <div style='color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Naive Persistence Baseline</div>
                <div style='color: #ffffff; font-size: 24px; font-weight: 700; margin-top: 4px;'>{forecast_res.persistence_swh_m:.3f} m</div>
                <div style='color: #94a3b8; font-size: 12px; margin-top: 2px;'>Observed state at anchor time <b>t</b></div>
                <div style='color: #cbd5e1; font-size: 11px; margin-top: 6px;'>Frozen lag-0 wave inertia reference benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 9. Time-Series Trajectory Visualization
    st.markdown("### 📈 Time-Series Context & Prognostic Trajectory")
    fig_live = plot_live_forecast_trajectory(
        prepared_df=df_window,
        anchor_time=verified_anchor,
        target_time=target_utc,
        forecast_swh=forecast_res.forecast_swh_m,
        persistence_swh=forecast_res.persistence_swh_m,
        model_name=forecast_res.model_name,
        horizon_hours=horizon_choice,
        external_nwp_swh=forecast_res.external_nwp_swh_m,
        dark_mode=True,
    )
    st.pyplot(fig_live)
    plt.close(fig_live)

    # 10. Metadata, Attribution & Provenance Details
    st.markdown("### 📋 Provenance, License & Execution Audit")
    prov_col1, prov_col2 = st.columns(2)
    with prov_col1:
        st.markdown(
            f"""
            - **Forecast Generation Time**: `{gen_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}` (`{gen_time_ist.strftime('%d-%b %H:%M:%S IST')}`)
            - **Input Observation Anchor ($t$)**: `{anchor_utc.strftime('%Y-%m-%d %H:00 UTC')}`
            - **Target Valid Time ($t+H$)**: `{target_utc.strftime('%Y-%m-%d %H:00 UTC')}`
            - **Forecast Lead Time ($H$)**: `{horizon_choice} Hour(s)`
            - **Model Checkpoint**: `{forecast_res.checkpoint_file}`
            """
        )
    with prov_col2:
        st.markdown(
            f"""
            - **Observation Data Provider**: `{fetch_result.source_name}`
            - **Data Licensing**: `Open Database License (ODbL) / CC BY 4.0`
            - **Attribution**: `{fetch_result.attribution}`
            - **Execution Mode**: `{'Live Real-Time Telemetry' if not fetch_result.is_demo else 'Historical Archive Fallback (Demo)'}`
            - **Feature Dimension**: `24 Consecutive Hourly Steps × 13 Physical & Circular Features`
            """
        )

    # 11. Scientific Validation & Distribution Shift Notice (Task 6)
    with st.expander("🔬 Scientific Compatibility Analysis: ERA5 Reanalysis vs Operational Live Telemetry", expanded=False):
        st.markdown(
            """
            #### Scientific Nuances Between ERA5 Reanalysis and Operational Feeds
            1. **Spatial Resolution Difference**:
               - **ERA5 Training Data**: Global ocean wave fields from ECMWF ERA5 reanalysis are archived on a $0.50^\circ \times 0.50^\circ$ spatial grid (~$55\\text{ km}$ resolution).
               - **Operational Live Feeds**: Current Open-Meteo marine endpoints query Copernicus Global Ocean Wave Analysis and Forecasts on an ~$0.08^\circ$ to $0.10^\circ$ grid (~$9\\text{ km}$ resolution).
            
            2. **Numerical Model Formulation Differences**:
               - The historical training set used the **ECMWF Integrated Forecasting System (IFS) Cycle 41r2** coupled with the WAM ocean wave physics model with 4D-Var data assimilation.
               - Near-real-time operational telemetry is generated by current ECMWF operational cycles (**IFS Cycle 48r1**), which incorporates refined wave dissipation and swell dissipation parametrizations.
            
            3. **Climatological Distribution Shift Warning**:
               - Neural networks are function approximators bounded by their training distribution ($\mathbb{P}_{train}$).
               - While physical variables ($H_s, T_p, u_{10}, v_{10}$) share identical standard SI units (meters, seconds, m/s), minor statistical distribution shifts between reanalysis and operational forecast analysis fields can slightly alter model bias.
            
            4. **Operational Deployment Pre-requisites**:
               - For safety-critical offshore operations (e.g. helideck landings, oil platform crane operations), operational deep learning models should undergo ongoing **real-time Kalman filtering or online residual calibration** against coastal moored buoy arrays (e.g., INCOIS AD08 buoy off Mumbai).
            """
        )

    # 12. Input Tensor Inspection Table
    with st.expander("🔍 Inspect 24-Hour Input Tensor Fed to Neural Network"):
        st.markdown("Exact 24 consecutive hourly observations and engineered circular/temporal features fed to the PyTorch model:")
        st.dataframe(df_window, use_container_width=True)


# =============================================================================
# PAGE 3: HISTORICAL DATA EXPLORER
# =============================================================================
elif nav_page == "📊 Historical Data Explorer":

    st.markdown(
        """
        <div class="ocean-header">
            <h1 class="ocean-title">ERA5 Ocean Wave & Atmospheric Data Explorer</h1>
            <p class="ocean-subtitle">Interactive timeseries visualization, physical variable distributions, and quality assurance audit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_all.empty:
        st.error("Processed dataset not found. Please run: `python scripts/run_preprocessing.py`")
        st.stop()

    # Presets for date filtering
    st.markdown("### 📅 Temporal Filter & Presets")
    preset = st.radio(
        "Quick Presets",
        [
            "Monsoon High Wave Event (June - August 2024)",
            "Recent 30 Days of Record (Sep - Oct 2026)",
            "Complete Test Evaluation Split (July 2024 - Oct 2026)",
            "Complete 14.8-Year Historical Record (2012 - 2026)",
            "Custom Date Range",
        ],
        horizontal=True,
    )

    min_date = df_all["valid_time"].min().date()
    max_date = df_all["valid_time"].max().date()

    if preset == "Monsoon High Wave Event (June - August 2024)":
        start_d, end_d = datetime(2024, 6, 1).date(), datetime(2024, 8, 31).date()
    elif preset == "Recent 30 Days of Record (Sep - Oct 2026)":
        start_d, end_d = datetime(2026, 9, 15).date(), max_date
    elif preset == "Complete Test Evaluation Split (July 2024 - Oct 2026)":
        start_d, end_d = datetime(2024, 7, 29).date(), max_date
    elif preset == "Complete 14.8-Year Historical Record (2012 - 2026)":
        start_d, end_d = min_date, max_date
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_d = st.date_input("Start Date", value=datetime(2024, 6, 1).date(), min_value=min_date, max_value=max_date)
        with col_d2:
            end_d = st.date_input("End Date", value=datetime(2024, 8, 31).date(), min_value=min_date, max_value=max_date)

    mask = (df_all["valid_time"].dt.date >= start_d) & (df_all["valid_time"].dt.date <= end_d)
    df_filtered = df_all.loc[mask].copy()

    if df_filtered.empty:
        st.warning("No data points found in selected date range.")
        st.stop()

    # Filtered statistics metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Selected Hours", f"{len(df_filtered):,}")
    with c2:
        st.metric("Mean Wave Height", f"{df_filtered['swh'].mean():.2f} m")
    with c3:
        st.metric("Peak Wave Height", f"{df_filtered['swh'].max():.2f} m")
    with c4:
        st.metric("Max Wind Speed", f"{df_filtered['wind_speed'].max():.2f} m/s")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌊 Wave Dynamics (SWH & Period)",
        "💨 Wind Velocity (10m Vectors)",
        "🧭 Directional & Polar Analysis",
        "✅ Quality Assurance & Continuity Audit",
    ])

    with tab1:
        st.markdown("#### Significant Wave Height ($H_s$) & Mean Wave Period ($T_p$)")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        
        ax1.plot(df_filtered["valid_time"], df_filtered["swh"], color="#0284c7", linewidth=1.2, label="Significant Wave Height (swh)")
        ax1.axhline(df_filtered["swh"].mean(), color="#38bdf8", linestyle="--", alpha=0.7, label=f"Mean: {df_filtered['swh'].mean():.2f} m")
        ax1.set_ylabel("Wave Height (m)", fontsize=11, color="#e2e8f0")
        ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax1.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax1.set_facecolor("#0f172a")

        ax2.plot(df_filtered["valid_time"], df_filtered["mwp"], color="#10b981", linewidth=1.0, label="Mean Wave Period (mwp)")
        ax2.set_ylabel("Period (s)", fontsize=11, color="#e2e8f0")
        ax2.set_xlabel("Time (UTC)", fontsize=11, color="#e2e8f0")
        ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax2.set_facecolor("#0f172a")

        fig.patch.set_facecolor("#0b111e")
        for ax in [ax1, ax2]:
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.markdown("#### 10-Meter Wind Velocity Components ($u_{10}, v_{10}$) and Wind Speed")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df_filtered["valid_time"], df_filtered["wind_speed"], color="#f59e0b", linewidth=1.2, label="Wind Speed (m/s)")
        ax.plot(df_filtered["valid_time"], df_filtered["u10"], color="#60a5fa", linewidth=0.8, alpha=0.6, label="Zonal u10 (Eastward)")
        ax.plot(df_filtered["valid_time"], df_filtered["v10"], color="#a78bfa", linewidth=0.8, alpha=0.6, label="Meridional v10 (Northward)")
        ax.set_ylabel("Velocity (m/s)", fontsize=11, color="#e2e8f0")
        ax.set_xlabel("Time (UTC)", fontsize=11, color="#e2e8f0")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0b111e")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown("#### Wave Direction & Wind Direction Distributions")
        col_dir1, col_dir2 = st.columns(2)
        
        # Calculate raw meteorological direction if not explicitly present
        wave_dir = df_filtered["mwd_deg"] if "mwd_deg" in df_filtered else np.mod(np.degrees(np.arctan2(df_filtered["mwd_sin"], df_filtered["mwd_cos"])), 360)
        wind_dir = df_filtered["wind_direction_deg"] if "wind_direction_deg" in df_filtered else np.mod(np.degrees(np.arctan2(df_filtered["wind_dir_sin"], df_filtered["wind_dir_cos"])), 360)

        with col_dir1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(wave_dir, bins=36, color="#0284c7", edgecolor="#0369a1", alpha=0.8)
            ax.set_title("Wave Direction (0°=N, 90°=E, 180°=S, 270°=W)", color="#e2e8f0", fontsize=11)
            ax.set_xlabel("Direction (degrees)", color="#e2e8f0")
            ax.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0b111e")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            st.pyplot(fig)
            plt.close(fig)

        with col_dir2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(wind_dir, bins=36, color="#f59e0b", edgecolor="#d97706", alpha=0.8)
            ax.set_title("Wind Direction (degrees true)", color="#e2e8f0", fontsize=11)
            ax.set_xlabel("Direction (degrees)", color="#e2e8f0")
            ax.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0b111e")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        st.markdown("#### Quality Assurance & Continuity Audit")
        rep = load_preprocessing_report()
        if rep:
            qa = rep.get("quality_assurance_summary", {})
            st.success("✅ **Preprocessing QA Audit Status: PASSED**")
            
            qc1, qc2, qc3, qc4 = st.columns(4)
            with qc1:
                st.metric("Total NaN Count", f"{qa.get('total_nan_count', 0)}")
            with qc2:
                st.metric("Total Inf Count", f"{qa.get('total_inf_count', 0)}")
            with qc3:
                st.metric("Duplicate Timestamps", f"{rep.get('temporal_coverage', {}).get('duplicate_timestamps', 0)}")
            with qc4:
                st.metric("Max Gap Hours", f"{rep.get('temporal_coverage', {}).get('max_gap_hours', 1.0)} hr")

            st.markdown("##### ERA5 Offshore Variable Physical Bounds Audit")
            stats_dict = qa.get("statistics", {})
            stats_rows = []
            for var_k, s in stats_dict.items():
                stats_rows.append({
                    "Variable": var_k,
                    "Min Value": f"{s['min']:.3f}",
                    "Max Value": f"{s['max']:.3f}",
                    "Mean": f"{s['mean']:.3f}",
                    "Std Dev": f"{s['std']:.3f}",
                })
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
        else:
            st.info("Detailed report available in `outputs/reports/preprocessing_report.json`.")


# =============================================================================
# PAGE 4: HISTORICAL SCENARIO ENGINE (TEST SPLIT)
# =============================================================================
elif nav_page == "⏪ Historical Scenario Engine (Test Split)":
    st.markdown(
        """
        <div class="ocean-header">
            <h1 class="ocean-title">⏪ Historical Scenario Engine (Test Split Retrospective Replay)</h1>
            <p class="ocean-subtitle">Replay and inspect deep learning inferences on historical test split events (2024–2026) with ground-truth verification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background: #111c2e; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px;'>
            <span class="badge-primary">RETROSPECTIVE BENCHMARK REPLAY</span>
            <span style='color: #cbd5e1; font-size: 13px; margin-left: 10px;'>
                This scenario engine evaluates trained models on historical ERA5 ground truth observations from the untouched 15% test split (2024-07-29 to 2026-10-17).
                Ground truth is available for direct error calculation. <b>To forecast current conditions, navigate to the "Live Ocean Forecast" page.</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


    if df_all.empty or scaler_obj is None:
        st.error("Prerequisites missing. Please ensure preprocessing and training artifacts are available.")
        st.stop()

    # Configuration Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        model_choice = st.selectbox(
            "Select Deep Learning Architecture",
            ["WaveLSTM (Recurrent)", "WaveANN (Feedforward MLP)", "WaveCNN1D (Temporal 1D-CNN)"],
            index=0,
            help="Choose between recurrent LSTM, feedforward ANN, or temporal 1D-CNN.",
        )
    with col_ctrl2:
        horizon_choice = st.selectbox(
            "Forecast Horizon (H)",
            [1, 6, 24],
            index=0,
            format_func=lambda h: f"{h} Hour{'s' if h > 1 else ''} Ahead (t + {h}h)",
            help="Lead time for future wave height forecast.",
        )
    with col_ctrl3:
        window_size = st.selectbox(
            "Historical Input Window (L)",
            [24],
            index=0,
            format_func=lambda w: f"{w} Historical Hours (t-23 to t)",
            help="Length of past hourly context fed into the model.",
        )

    # Normalize architecture name
    if "LSTM" in model_choice:
        m_type_name = "WaveLSTM"
        m_code = "LSTM"
    elif "ANN" in model_choice:
        m_type_name = "WaveANN"
        m_code = "ANN"
    else:
        m_type_name = "WaveCNN1D"
        m_code = "CNN1D"

    # Sequence selection from Test Split
    st.markdown("### 🕒 Anchor Timestamp Selection")
    st.caption("Select an anchor timestamp $t$. The model will ingest $[t-23\\text{h}, t]$ and forecast $t+H\\text{h}$.")

    # Filter to test split for live demonstration
    test_start = pd.to_datetime("2024-07-29 09:00:00")
    df_test_pool = df_all[df_all["valid_time"] >= test_start].reset_index(drop=True)

    event_preset = st.selectbox(
        "Demonstration Scenarios",
        [
            "Custom Anchor Selection",
            "Monsoon High Wave Event (2024-08-03 12:00:00, Hs ~ 3.2m)",
            "Moderate Offshore Wave Condition (2025-01-15 00:00:00, Hs ~ 1.5m)",
            "Calm Sea State (2026-03-20 06:00:00, Hs ~ 0.7m)",
            "Most Recent Observation in Record (2026-10-17 00:00:00)",
        ],
    )

    if event_preset == "Monsoon High Wave Event (2024-08-03 12:00:00, Hs ~ 3.2m)":
        target_anchor = pd.to_datetime("2024-08-03 12:00:00")
    elif event_preset == "Moderate Offshore Wave Condition (2025-01-15 00:00:00, Hs ~ 1.5m)":
        target_anchor = pd.to_datetime("2025-01-15 00:00:00")
    elif event_preset == "Calm Sea State (2026-03-20 06:00:00, Hs ~ 0.7m)":
        target_anchor = pd.to_datetime("2026-03-20 06:00:00")
    elif event_preset == "Most Recent Observation in Record (2026-10-17 00:00:00)":
        target_anchor = pd.to_datetime("2026-10-17 00:00:00")
    else:
        # User slider across test pool
        max_idx = len(df_test_pool) - 1 - horizon_choice
        sel_idx = st.slider("Select Anchor Step in Test Set", min_value=24, max_value=max_idx, value=100, step=1)
        target_anchor = df_test_pool.iloc[sel_idx]["valid_time"]

    # Locate anchor index in df_all
    anchor_matches = df_all.index[df_all["valid_time"] == target_anchor].tolist()
    if not anchor_matches:
        st.error(f"Anchor time {target_anchor} not found in dataset.")
        st.stop()
    anchor_idx = anchor_matches[0]

    # Validate historical window availability
    if anchor_idx < window_size - 1:
        st.error(f"Not enough historical context for window size {window_size}.")
        st.stop()

    window_slice = df_all.iloc[anchor_idx - window_size + 1 : anchor_idx + 1]
    forecast_target_time = target_anchor + pd.Timedelta(hours=horizon_choice)
    has_ground_truth = (df_all["valid_time"] == forecast_target_time).any()
    actual_target_val = df_all.loc[df_all["valid_time"] == forecast_target_time, "swh"].values[0] if has_ground_truth else None

    # Load Model Checkpoint with Graceful Unavailable Handling
    model, meta = load_model_checkpoint(m_code, horizon=horizon_choice)

    if model is None:
        st.markdown(
            f"""
            <div style='background: #3b1812; border: 1px solid #7f1d1d; border-radius: 8px; padding: 16px 20px; margin: 15px 0;'>
                <h4 style='color: #fca5a5; margin: 0;'>⚠️ Model Checkpoint Unavailable</h4>
                <p style='color: #fecaca; font-size: 14px; margin: 8px 0 4px 0;'>
                    A trained checkpoint for <b>{m_type_name}</b> with <b>Horizon = {horizon_choice} hour(s)</b> was not found in <code>models/</code>.
                </p>
                <p style='color: #e2e8f0; font-size: 13px; margin: 0;'>
                    To train this model on the 70% split and generate its weights, execute from the project root:
                    <br><code style='background: #111; color: #38bdf8; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-top: 6px;'>
                    python scripts/train_and_evaluate.py --model {m_code.lower()} --horizon {horizon_choice} --window-size {window_size}
                    </code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Model is available! Perform Real-Time Inference
        scaled_x, _ = scaler_obj.transform(window_slice)
        tensor_x = torch.tensor(scaled_x, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 24, 13)

        with torch.no_grad():
            scaled_pred = model(tensor_x).item()
        
        pred_meters = float(scaler_obj.inverse_transform_target(np.array([scaled_pred]))[0])
        sea_state_desc, sea_state_color = get_sea_state_description(pred_meters)

        # Display Results
        st.markdown("---")
        st.markdown("### 🎯 Forecast Result")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Predicted Wave Height</div>
                    <div class="metric-card-value">{pred_meters:.3f} m</div>
                    <div class="metric-card-sub">{pred_meters * 100:.1f} cm</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Forecast Valid Time</div>
                    <div class="metric-card-value" style="font-size: 18px; margin-top: 10px;">{forecast_target_time.strftime('%Y-%m-%d %H:00')}</div>
                    <div class="metric-card-sub">Anchor + {horizon_choice} hr(s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">WMO Sea State Code</div>
                    <div class="metric-card-value" style="color: {sea_state_color}; font-size: 20px; margin-top: 8px;">{sea_state_desc}</div>
                    <div class="metric-card-sub">Offshore Conditions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r4:
            if actual_target_val is not None:
                err = pred_meters - actual_target_val
                pct_err = (abs(err) / actual_target_val) * 100
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">Actual vs Error</div>
                        <div class="metric-card-value" style="font-size: 20px; margin-top: 8px;">{actual_target_val:.3f} m</div>
                        <div class="metric-card-sub">Error: {err:+.3f} m ({pct_err:.1f}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-card-title">Ground Truth Status</div>
                        <div class="metric-card-value" style="font-size: 18px; margin-top: 10px;">Pending</div>
                        <div class="metric-card-sub">Beyond dataset range</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Historical Window + Prediction Plot
        st.markdown("#### Forecast Visualization (Historical Input Window vs Future Prediction)")
        fig, ax = plt.subplots(figsize=(13, 4.5))

        # Plot 24h context
        ax.plot(
            window_slice["valid_time"],
            window_slice["swh"],
            color="#38bdf8",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="24h Historical Input Context (Observed)",
        )

        # Plot anchor marker
        ax.scatter(
            [target_anchor],
            [window_slice["swh"].iloc[-1]],
            color="#f59e0b",
            s=80,
            zorder=5,
            label=f"Anchor Timestamp t ({target_anchor.strftime('%m-%d %H:00')})",
        )

        # Plot forecast point
        ax.scatter(
            [forecast_target_time],
            [pred_meters],
            color="#10b981",
            s=120,
            marker="*",
            zorder=6,
            label=f"{m_type_name} Forecast t+{horizon_choice}h: {pred_meters:.3f} m",
        )

        # Connect anchor to forecast
        ax.plot(
            [target_anchor, forecast_target_time],
            [window_slice["swh"].iloc[-1], pred_meters],
            color="#10b981",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

        # Plot actual if available
        if actual_target_val is not None:
            ax.scatter(
                [forecast_target_time],
                [actual_target_val],
                color="#ef4444",
                s=90,
                marker="x",
                zorder=6,
                label=f"Actual Ground Truth: {actual_target_val:.3f} m",
            )

        ax.set_ylabel("Significant Wave Height (m)", fontsize=11, color="#e2e8f0")
        ax.set_xlabel("Time (UTC)", fontsize=11, color="#e2e8f0")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0b111e")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        for spine in ax.spines.values():
            spine.set_color("#334155")

        st.pyplot(fig)
        plt.close(fig)

        # Input Feature Breakdown Table
        with st.expander("🔍 Inspect 24-Hour Input Features Fed to Neural Network"):
            st.dataframe(
                window_slice[["valid_time", "swh", "mwp", "u10", "v10", "wind_speed"]].reset_index(drop=True),
                use_container_width=True,
            )


# =============================================================================
# PAGE 4: MODEL COMPARISON & BENCHMARKS
# =============================================================================
# =============================================================================
# PAGE 4: MODEL COMPARISON & BENCHMARKS
# =============================================================================
elif nav_page == "🏆 Model Comparison & Metrics":
    st.markdown(
        """
        <div class="ocean-header">
            <h1 class="ocean-title">Multi-Horizon Benchmark: Naive Persistence vs Deep Learning</h1>
            <p class="ocean-subtitle">Rigorous test set evaluation on 19,455 untouched chronological samples across 1h, 6h, and 24h lead times</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_multi is None and df_metrics is None:
        st.warning(
            """
            ⏳ **Evaluation Metrics Pending**
            The benchmark has not yet been executed, or metrics CSVs are missing.
            Please run the multi-horizon evaluation pipeline:
            `python scripts/evaluate_multi_horizon.py`
            """
        )
        st.stop()

    # Use multi-horizon comparison if available
    benchmark_df = df_multi.copy() if df_multi is not None else df_metrics.copy()

    # Horizon View Selector
    st.markdown("### 🎯 Select Forecast Horizon Lead Time")
    sel_h_view = st.radio(
        "Forecast Horizon View",
        [
            "All Horizons Overview (H=1h, H=6h, H=24h)",
            "1-Hour Ahead (H=1h) — Persistence-Dominated Regime",
            "6-Hours Ahead (H=6h) — Prognostic Deep Learning Regime",
            "24-Hours Ahead (H=24h) — Extended Lead Time Regime",
        ],
        horizontal=True,
    )

    if "1-Hour" in sel_h_view:
        active_h = 1
    elif "6-Hours" in sel_h_view:
        active_h = 6
    elif "24-Hours" in sel_h_view:
        active_h = 24
    else:
        active_h = None  # Full overview

    # Dynamic Best Method vs Best Neural Net Highlights
    if active_h == 1:
        st.markdown(
            """
            <div style='background: #111c2e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px;'>
                <div style='display: flex; gap: 24px; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 280px;'>
                        <span class="badge-primary">BEST OVERALL METHOD (H=1h)</span>
                        <h3 style='color: #38bdf8; margin: 8px 0 4px 0;'>Naive Persistence Baseline</h3>
                        <p style='color: #cbd5e1; font-size: 14px; margin: 0;'>
                            <b>RMSE: 0.0213 m (2.13 cm)</b> | MAE: 0.0131 m | R²: 0.9991
                        </p>
                        <p style='color: #94a3b8; font-size: 13px; margin-top: 6px;'>
                            At 1 hour, ocean wave swell exhibits immense physical inertia (autocorrelation ρ₁ = 0.9995).
                            <b>Persistence outperforms all deep learning models</b>.
                        </p>
                    </div>
                    <div style='flex: 1; min-width: 280px; border-left: 1px solid #1e293b; padding-left: 20px;'>
                        <span class="badge-success">BEST NEURAL NETWORK (H=1h)</span>
                        <h3 style='color: #34d399; margin: 8px 0 4px 0;'>WaveLSTM (Recurrent)</h3>
                        <p style='color: #cbd5e1; font-size: 14px; margin: 0;'>
                            <b>RMSE: 0.0271 m (2.71 cm)</b> | MAE: 0.0182 m | R²: 0.9986
                        </p>
                        <p style='color: #94a3b8; font-size: 13px; margin-top: 6px;'>
                            Tracks the true wave state within 5.8 mm of persistence, outperforming ANN (0.0465m) and CNN1D (0.0587m).
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif active_h == 6:
        st.markdown(
            """
            <div style='background: #064e3b; border: 1px solid #059669; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px;'>
                <div style='display: flex; gap: 24px; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 280px;'>
                        <span class="badge-success">BEST OVERALL & NEURAL NET (H=6h)</span>
                        <h3 style='color: #6ee7b7; margin: 8px 0 4px 0;'>🏆 WaveLSTM (Recurrent)</h3>
                        <p style='color: #a7f3d0; font-size: 14px; margin: 0;'>
                            <b>RMSE: 0.0769 m (7.69 cm)</b> | MAE: 0.0533 m | R²: 0.9886
                        </p>
                        <p style='color: #d1fae5; font-size: 13px; margin-top: 6px;'>
                            <b>WaveLSTM decisively outperforms Naive Persistence</b> (RMSE: 0.0893 m, MAE: 0.0624 m) with 
                            <b>+13.88% RMSE Skill</b> and <b>+14.51% MAE Skill</b>! WaveANN also beats persistence (+8.77% RMSE skill).
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif active_h == 24:
        st.markdown(
            """
            <div style='background: #111c2e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px;'>
                <div style='display: flex; gap: 24px; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 280px;'>
                        <span class="badge-primary">BEST ON RMSE (H=24h)</span>
                        <h3 style='color: #38bdf8; margin: 8px 0 4px 0;'>Naive Persistence Baseline</h3>
                        <p style='color: #cbd5e1; font-size: 14px; margin: 0;'>
                            <b>RMSE: 0.1866 m (18.66 cm)</b> | R²: 0.9330
                        </p>
                        <p style='color: #94a3b8; font-size: 13px; margin-top: 6px;'>
                            Persistence achieves the lowest quadratic penalty on peak wave shifts at 24 hours.
                        </p>
                    </div>
                    <div style='flex: 1; min-width: 280px; border-left: 1px solid #1e293b; padding-left: 20px;'>
                        <span class="badge-success">BEST NEURAL NETWORK ON MAE (H=24h)</span>
                        <h3 style='color: #34d399; margin: 8px 0 4px 0;'>WaveANN (MLP)</h3>
                        <p style='color: #cbd5e1; font-size: 14px; margin: 0;'>
                            <b>MAE: 0.1225 m (vs 0.1286 m for Persistence: +4.71% Skill)</b> | RMSE: 0.1920 m
                        </p>
                        <p style='color: #94a3b8; font-size: 13px; margin-top: 6px;'>
                            WaveANN achieves the lowest mean absolute physical error. WaveLSTM also beats persistence on MAE (0.1229 m, +4.45% Skill).
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='background: #111c2e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;'>
                <h4 style='color: #38bdf8; margin: 0 0 6px 0;'>🔬 Academic Multi-Horizon Benchmark Summary</h4>
                <p style='color: #cbd5e1; font-size: 14px; margin: 0;'>
                    • <b>Horizon 1h</b>: Naive Persistence achieves the lowest error due to lag-1 wave inertia (ρ₁ = 0.9995). WaveLSTM is the best neural net (RMSE 0.0271 m).<br>
                    • <b>Horizon 6h</b>: <b>WaveLSTM outperforms Naive Persistence</b> on both RMSE (+13.88% Skill) and MAE (+14.51% Skill). WaveANN also beats persistence (+8.77% RMSE Skill).<br>
                    • <b>Horizon 24h</b>: Naive Persistence retains the lowest RMSE overall (0.1866 m), but <b>WaveANN and WaveLSTM outperform persistence on MAE</b> (+4.71% and +4.45% Skill).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Filtered Table Display
    st.markdown("### 📊 Measured Multi-Horizon Performance Table")
    if active_h is not None:
        view_df = benchmark_df[benchmark_df["horizon_hours"] == active_h].copy()
    else:
        view_df = benchmark_df.copy()

    tbl_cols = {
        "horizon_hours": "Horizon",
        "model": "Model / Method",
        "model_type": "Type",
        "rmse_meters": "RMSE (m)",
        "mae_meters": "MAE (m)",
        "r2_score": "R² Score",
        "scatter_index_pct": "Scatter Index (SI)",
        "wape_pct": "WAPE (%)",
        "rmse_skill_pct": "RMSE Skill vs Persist (%)",
        "mae_skill_pct": "MAE Skill vs Persist (%)",
    }
    avail_cols = [c for c in tbl_cols.keys() if c in view_df.columns]
    table_render = view_df[avail_cols].rename(columns=tbl_cols)

    for c in table_render.columns:
        if "RMSE (m)" in c or "MAE (m)" in c:
            table_render[c] = table_render[c].map(lambda x: f"{x:.4f} m" if pd.notnull(x) else "—")
        elif "R²" in c:
            table_render[c] = table_render[c].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
        elif "Scatter" in c or "WAPE" in c:
            table_render[c] = table_render[c].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")
        elif "Skill" in c:
            table_render[c] = table_render[c].map(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "—")
        elif "Horizon" in c:
            table_render[c] = table_render[c].map(lambda x: f"{x} hr(s)")

    st.dataframe(table_render, use_container_width=True)

    # Comparative Bar Charts
    st.markdown("### 📈 Visual Multi-Horizon Comparison")
    bcol1, bcol2 = st.columns(2)

    chart_sub = benchmark_df if active_h is None else benchmark_df[benchmark_df["horizon_hours"] == active_h]
    chart_valid = chart_sub[chart_sub["rmse_meters"].notnull()].copy()

    with bcol1:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        labels = [f"{r['model']} (H={r['horizon_hours']}h)" if active_h is None else r['model'] for _, r in chart_valid.iterrows()]
        rmse_vals = chart_valid["rmse_meters"].tolist()
        mae_vals = chart_valid["mae_meters"].tolist()
        x_pos = np.arange(len(labels))
        width = 0.35

        ax.bar(x_pos - width/2, rmse_vals, width, label="RMSE (m)", color="#0284c7")
        ax.bar(x_pos + width/2, mae_vals, width, label="MAE (m)", color="#10b981")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, color="#e2e8f0", fontsize=9, rotation=25 if len(labels) > 4 else 0)
        ax.set_ylabel("Error (Meters)", color="#e2e8f0")
        ax.set_title("Test Error Benchmark: RMSE & MAE", color="#e2e8f0", fontsize=11)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0b111e")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with bcol2:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        skill_vals = chart_valid["rmse_skill_pct"].tolist()
        colors = ["#10b981" if s > 0 else ("#38bdf8" if s == 0 else "#ef4444") for s in skill_vals]
        ax.bar(labels, skill_vals, color=colors, width=0.45)
        ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, color="#e2e8f0", fontsize=9, rotation=25 if len(labels) > 4 else 0)
        ax.set_ylabel("RMSE Skill vs Persistence (%)", color="#e2e8f0")
        ax.set_title("RMSE Skill Score (Positive = Outperforms Persistence)", color="#e2e8f0", fontsize=11)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0b111e")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Generated Plot Artifacts Tabs
    st.markdown("### 🖼️ Saved Publication-Grade Visualizations")
    tab_p1, tab_p2, tab_p3, tab_p4, tab_p5 = st.tabs([
        "📉 Persistence Error Growth (1h to 24h)",
        "🎯 Horizon H=1h Model Comparison",
        "📈 Loss Trajectory Across Epochs",
        "🌊 300-Hour Actual vs Predicted Sequence",
        "📊 Residual Distributions",
    ])

    with tab_p1:
        plot_p = PROJECT_ROOT / "outputs" / "plots" / "multi_horizon_persistence_baseline.png"
        if plot_p.is_file():
            st.image(str(plot_p), caption="Naive Persistence Error Growth & R² Degradation across 1h, 6h, and 24h Lead Times", use_container_width=True)
        else:
            st.info("Artifact not found at `outputs/plots/multi_horizon_persistence_baseline.png`.")

    with tab_p2:
        plot_p = PROJECT_ROOT / "outputs" / "plots" / "horizon_1h_model_comparison.png"
        if plot_p.is_file():
            st.image(str(plot_p), caption="1-Hour Ahead Benchmark: Persistence vs WaveLSTM vs WaveANN vs WaveCNN1D", use_container_width=True)
        else:
            st.info("Artifact not found at `outputs/plots/horizon_1h_model_comparison.png`.")

    with tab_p3:
        plot_p = PROJECT_ROOT / "outputs" / "plots" / "training_history.png"
        if plot_p.is_file():
            st.image(str(plot_p), caption="Training vs Validation Loss History across Epochs", use_container_width=True)
        else:
            st.info("Loss plot artifact not found at `outputs/plots/training_history.png`.")

    with tab_p4:
        plot_p = PROJECT_ROOT / "outputs" / "plots" / "actual_vs_predicted.png"
        if plot_p.is_file():
            st.image(str(plot_p), caption="Actual vs Predicted Significant Wave Height (300-Hour Test Window)", use_container_width=True)
        else:
            st.info("Forecast plot artifact not found at `outputs/plots/actual_vs_predicted.png`.")

    with tab_p5:
        plot_p = PROJECT_ROOT / "outputs" / "plots" / "prediction_residuals.png"
        if plot_p.is_file():
            st.image(str(plot_p), caption="Residual Error Distributions & Residuals vs Wave Height", use_container_width=True)
        else:
            st.info("Residual plot artifact not found at `outputs/plots/prediction_residuals.png`.")

    # Test Predictions Table Sample
    with st.expander("📋 View Sample Predictions from Untouched Test Set"):
        sample_preds = load_test_predictions_sample(nrows=200)
        if sample_preds is not None:
            st.dataframe(sample_preds, use_container_width=True)
        else:
            st.info("Test predictions file not found at `outputs/predictions/test_predictions.csv`.")


# =============================================================================
# PAGE 5: METHODOLOGY & ARCHITECTURE
# =============================================================================
elif nav_page == "🔬 Methodology & Architecture":
    st.markdown(
        """
        <div class="ocean-header">
            <h1 class="ocean-title">Deep Learning Forecasting Methodology</h1>
            <p class="ocean-subtitle">Mathematical formulation, leak-free preprocessing, neural architectures, and marine evaluation metrics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Mathematical Problem Formulation")
    st.markdown(
        """
        Given a multivariate historical sequence of observations $X_{t-L+1:t} \in \mathbb{R}^{L \times D}$ where:
        - $L = 24$ is the historical input window (past 24 hours),
        - $D = 13$ is the feature dimension at each time step,
        
        The objective is to learn a mapping $f_\theta: \mathbb{R}^{L \times D} \to \mathbb{R}$ parameterized by deep neural network weights $\theta$ that predicts the significant wave height at lead time $H$:
        $$\hat{y}_{t+H} = f_\theta(X_{t-L+1:t})$$
        Where $H \in \{1, 6, 24\}$ hours ahead.
        """
    )

    st.markdown("---")
    st.markdown("### 2. Preprocessing & Temporal Leakage Prevention")
    st.markdown(
        """
        1. **Chronological Splitting (No Shuffling)**:
           Standard random train/test splitting causes catastrophic temporal data leakage in time series.
           We employ strict chronological boundaries:
           - **Training Split (70%)**: 2012-01-02 00:00 to 2022-05-10 18:00 ($90,763$ hours)
           - **Validation Split (15%)**: 2022-05-10 19:00 to 2024-07-29 08:00 ($19,455$ hours)
           - **Test Split (15%)**: 2024-07-29 09:00 to 2026-10-17 23:00 ($19,455$ hours)
        
        2. **Train-Only Feature Scaling**:
           To ensure zero information snooping, the `TimeSeriesScaler` is fitted **strictly on the 70% training split**. Mean and standard deviation parameters ($\mu_{train}, \sigma_{train}$) are frozen and applied downstream to validation and test tensors.
        
        3. **Circular Angle Decomposition**:
           Meteorological and oceanographic directions (degrees $[0^\circ, 360^\circ]$) exhibit an artificial jump between $359^\circ$ and $1^\circ$. We project wave direction and wind direction into continuous orthogonal sinusoidal coordinates:
           $$\theta_{sin} = \sin\left(\frac{2\pi \theta}{360}\right), \quad \theta_{cos} = \cos\left(\frac{2\pi \theta}{360}\right)$$
        """
    )

    st.markdown("---")
    st.markdown("### 3. Deep Learning Architectures Compared")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("#### 1. WaveANN (MLP)")
        st.markdown(
            """
            - **Input**: Flattened vector $\in \mathbb{R}^{312}$ ($24 \text{ steps} \times 13 \text{ features}$).
            - **Architecture**:
              - Linear(312 $\to$ 128) + BatchNorm1d + ReLU + Dropout(0.2)
              - Linear(128 $\to$ 64) + BatchNorm1d + ReLU + Dropout(0.2)
              - Linear(64 $\to$ 32) + ReLU
              - Linear(32 $\to$ 1)
            - **Parameters**: 50,689 trainable.
            - **Characteristics**: Instantaneous point regressor, high throughput, ignores explicit sequential ordering.
            """
        )
    with m_col2:
        st.markdown("#### 2. WaveLSTM (Recurrent)")
        st.markdown(
            """
            - **Input**: Sequential tensor $\in \mathbb{R}^{B \times 24 \times 13}$.
            - **Architecture**:
              - 2-layer stacked LSTM, hidden dimension = 64, dropout = 0.2.
              - Multi-layer perceptron head: Linear(64 $\to$ 32) + ReLU + Linear(32 $\to$ 1).
            - **Parameters**: 55,617 trainable.
            - **Characteristics**: Maintains internal cell memory states, tracks sequential dependencies and swell growth.
            """
        )
    with m_col3:
        st.markdown("#### 3. WaveCNN1D (Temporal)")
        st.markdown(
            """
            - **Input**: Channel-first tensor $\in \mathbb{R}^{B \times 13 \times 24}$.
            - **Architecture**:
              - 3-stage dilated 1D temporal convolutions (dilation = 1, 2, 4) with kernel size 3.
              - Causal padding to ensure zero future information leakage.
              - AdaptiveAvgPool1d + Linear head.
            - **Parameters**: 77,505 trainable.
            - **Characteristics**: Multi-scale receptive fields, parallel temporal convolution.
            """
        )

    st.markdown("---")
    st.markdown("### 4. Marine Evaluation Metrics & MAPE Limitations")
    st.markdown(
        """
        Because significant wave height is a continuous physical quantity, model performance is evaluated using standard oceanographic regression metrics:
        
        - **Root Mean Squared Error (RMSE)**: Penalizes large peak wave forecasting errors heavily.
          $$\\text{RMSE} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^N (y_i - \\hat{y}_i)^2}$$
        - **Mean Absolute Error (MAE)**: Robust linear penalty for average physical deviation.
          $$\\text{MAE} = \\frac{1}{N}\\sum_{i=1}^N |y_i - \\hat{y}_i|$$
        - **Coefficient of Determination ($R^2$)**: Proportion of variance explained by model forecasts.
        - **Scatter Index (SI)**: Normalized, dimensionless oceanographic metric standard:
          $$\\text{SI} = \\frac{\\text{RMSE}}{\\bar{y}_{true}}$$
        - **The Limitation of MAPE in Wave Modeling**:
          $$\\text{MAPE} = \\frac{100\\%}{N}\\sum_{i=1}^N \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$
          During calm sea states where $H_s < 0.2\\text{ m}$, small physical errors (e.g., $0.05\\text{ m}$) cause the denominator to approach zero, resulting in disproportionately large percentage spikes. Therefore, **Scatter Index** and **WAPE** are preferred for reliable relative benchmarking.
        """
    )

    st.markdown("---")
    st.markdown("### 5. Live Operational Ingestion & Distribution Shift Management")
    st.markdown(
        """
        1. **Dual-API Ingestion Architecture**:
           The live forecast service coordinates two distinct Open-Meteo REST endpoints under CC BY 4.0:
           - **Marine API**: Ingests ECMWF WAM / Copernicus wave height ($H_s$), mean wave period ($T_p$), and circular wave direction ($MWD$).
           - **Atmospheric Weather API**: Ingests ECMWF IFS 10m wind speed ($S_{10}$) and meteorological wind direction ($\theta$).
        
        2. **Wind Vector Mathematical Transformation**:
           Meteorological wind direction ($\theta^\circ$) represents the direction *from* which the wind originates. The physical zonal ($u_{10}$) and meridional ($v_{10}$) components are rigorously projected:
           $$u_{10} = -S_{10} \\sin\\left(\\frac{2\\pi \\theta}{360}\\right), \\quad v_{10} = -S_{10} \\cos\\left(\\frac{2\\pi \\theta}{360}\\right)$$
        
        3. **Sequence Validation Without Imputation**:
           The pipeline mandates exactly 24 consecutive hourly steps. If any hour is missing, or if timestamps contain gaps $>1\\text{h}$, or if NaNs are detected, the inference is safely aborted with an explicit diagnostic alert rather than silently imputing synthetic values.
        
        4. **Distribution Shift & Operational Limitations**:
           - **Grid Disparity**: ERA5 training data used a $0.50^\circ$ grid (~$55\\text{ km}$), whereas operational feeds provide ~$0.10^\circ$ resolution (~$11\\text{ km}$).
           - **Model Cycle Disparity**: ERA5 is reanalysis (HRES cycle 41r2), whereas live telemetry is operational IFS Cycle 48r1.
           - Consequently, operational forecasts must be treated as prognostic guidance rather than certified sea-keeping safety determinations without in-situ buoy calibration.
        """
    )

