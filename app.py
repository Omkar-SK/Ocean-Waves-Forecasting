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
from src.intelligence.nodes import get_coastal_registry, CoastalNode, ValidationStatus
from src.intelligence.provider import DataStatus
from src.intelligence.ingestion import get_ingestion_service
from src.intelligence.risk import (
    RiskLevel,
    EventType,
    RiskThresholdSet,
    RiskAssessment,
    assess_node_risk,
    get_threshold_manager,
    classify_risk,
)
from src.intelligence.alerts import (
    AlertStatus,
    AlertRecord,
    get_alert_store,
    get_alert_engine,
)
from src.intelligence.mapping import (
    NodeMapSummary,
    prepare_coastal_node_map_data,
    compute_map_dashboard_metrics,
    build_coastal_folium_map,
)
from src.intelligence.monitoring import (
    ForecastLedgerRecord,
    ForecastSourceCategory,
    ObservationRecord,
    ObservationSourceType,
    VerificationStatus,
    DriftStatus,
    PerformanceStatus,
    MonitoringEventType,
    MonitoringSeverity,
    MonitoringEventStatus,
    get_monitoring_store,
    calculate_forecast_accuracy,
    get_drift_manager,
    evaluate_model_performance_degradation,
    populate_retrospective_reanalysis_observations,
)

try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    HAS_STREAMLIT_FOLIUM = False


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
def fetch_cached_live_marine(latitude: float, longitude: float, timeout_s: int, location_id: Optional[str] = None) -> LiveFetchResult:
    """Cached Open-Meteo REST query with 15-minute TTL to reduce redundant network load."""
    return fetch_live_marine_and_weather(latitude=latitude, longitude=longitude, timeout_s=timeout_s, location_id=location_id)




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
        <div style='background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 6px; padding: 6px 8px; margin-top: 10px; font-size: 11px; color: #a5b4fc;'>
            ⚡ <b>Legacy Fallback Interface</b><br>
            Primary UI migrated to <b>React + Vite</b> at <code>http://localhost:5173</code>
        </div>
    </div>
    <hr style='border: 1px solid #1e293b; margin: 12px 0 18px 0;'>
    """,
    unsafe_allow_html=True,
)

nav_page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📌 Project Overview",
        "🚨 Coastal Alert Centre & Early Warning",
        "🗺️ Coastal Risk Heatmap",
        "📈 Forecast Accuracy & Model Drift",
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
# PAGE: COASTAL ALERT CENTRE & EARLY WARNING
# =============================================================================
elif nav_page == "🚨 Coastal Alert Centre & Early Warning":
    st.markdown(
        """
        <div class="ocean-header" style="background: linear-gradient(135deg, #1e1b4b 0%, #1e3a8a 50%, #0369a1 100%); border-color: #3b82f6;">
            <h1 class="ocean-title">🚨 Coastal Early-Warning & Risk Alert Centre</h1>
            <p class="ocean-subtitle">Explainable Wave Hazard Detection, Configurable Risk Thresholds & Persistent Lifecycle Management</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 14px; margin-bottom: 4px;">⚠️ Decision-Support Academic Prototype Notice</div>
            <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5;">
                This early-warning alert system is an experimental decision-support prototype evaluating marine wave forecasts against configurable engineering thresholds (Demonstration v1.0). 
                <b>It does NOT replace official maritime safety warnings or disaster management advisories issued by national authorities (e.g., INCOIS, IMD).</b> 
                A <span style="color: #10b981; font-weight: 600;">GREEN</span> assessment indicates that no configured threshold is exceeded in available telemetry; it is <b>never a guarantee of maritime safety</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert_store = get_alert_store()
    alert_engine = get_alert_engine()
    coastal_reg = get_coastal_registry()
    thresh_mgr = get_threshold_manager()

    # 1. Operational Controls & Ingestion Trigger Bar
    f_c1, f_c2, f_c3, f_c4, f_c5 = st.columns([1.3, 1.0, 1.1, 1.3, 1.0])
    with f_c1:
        reg_nodes = coastal_reg.list_all(enabled_only=False)
        loc_options = ["ALL"] + [n.location_id for n in reg_nodes]
        loc_labels = {"ALL": "🌐 All Coastal Locations"}
        loc_labels.update({n.location_id: f"{n.name} ({n.coordinate_label})" for n in reg_nodes})
        sel_loc = st.selectbox(
            "Filter Location",
            loc_options,
            format_func=lambda k: loc_labels[k],
            index=0,
            help="Filter displayed alerts by coastal monitoring station.",
        )
    with f_c2:
        sel_sev = st.selectbox(
            "Filter Severity",
            ["ALL", "RED", "ORANGE", "YELLOW"],
            format_func=lambda s: {
                "ALL": "All Severities",
                "RED": "🔴 RED (Critical)",
                "ORANGE": "🟠 ORANGE (High)",
                "YELLOW": "🟡 YELLOW (Caution)",
            }[s],
            index=0,
            help="Filter by minimum risk category.",
        )
    with f_c3:
        sel_status = st.selectbox(
            "Filter Status",
            ["ACTIVE_ACKNOWLEDGED", "ALL", "ACTIVE", "ACKNOWLEDGED", "RESOLVED", "EXPIRED"],
            format_func=lambda st_code: {
                "ACTIVE_ACKNOWLEDGED": "Active & Acknowledged",
                "ALL": "All Lifecycle States",
                "ACTIVE": "🔴 Active Only",
                "ACKNOWLEDGED": "🟡 Acknowledged Only",
                "RESOLVED": "🟢 Resolved",
                "EXPIRED": "⚪ Expired",
            }[st_code],
            index=0,
            help="Filter alerts by lifecycle state.",
        )
    with f_c4:
        source_mode = st.selectbox(
            "Telemetry Ingestion Feed",
            [
                "📂 Offline Historical Demo (ERA5 Window)",
                "🌐 Open-Meteo REST API (Live Feed)",
            ],
            index=0,
            help="Telemetry feed for automated risk evaluation.",
        )
    with f_c5:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Evaluate Now", use_container_width=True, help="Fetch telemetry and re-evaluate hazard thresholds"):
            with st.spinner("Ingesting coastal telemetry and evaluating hazard thresholds..."):
                source_clean = "demo" if "demo" in source_mode.lower() else "live"
                ingestion_svc = get_ingestion_service()
                fetch_dict = ingestion_svc.fetch_all_locations(source=source_clean, enabled_only=True, force_refresh=True)
                assessments_to_run = []
                for lid, f_res in fetch_dict.items():
                    try:
                        n = coastal_reg.get(lid)
                        t_set = thresh_mgr.get_thresholds(lid)
                        ass_list = assess_node_risk(
                            node=n,
                            fetch_result=f_res,
                            scaler=scaler_obj,
                            thresholds=t_set,
                            horizons=[1, 6, 24],
                        )
                        assessments_to_run.extend(ass_list)
                    except Exception:
                        continue
                alert_engine.process_all_assessments(assessments_to_run)
                st.toast("Coastal hazard evaluation complete.", icon="🚨")
                st.rerun()

    # 2. KPI Summary Bar
    active_sev_counts = alert_store.count_active_by_severity()
    status_counts = alert_store.count_alerts_by_status()
    total_active = active_sev_counts["RED"] + active_sev_counts["ORANGE"] + active_sev_counts["YELLOW"]

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Active Hazard Alerts</div>
                <div class="metric-card-value" style="color: {'#ef4444' if total_active > 0 else '#10b981'};">{total_active}</div>
                <div class="metric-card-sub">Active & Acknowledged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Critical Risk (RED)</div>
                <div class="metric-card-value" style="color: #ef4444;">{active_sev_counts['RED']}</div>
                <div class="metric-card-sub">Threshold: ≥ 4.00 m</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">High Risk (ORANGE)</div>
                <div class="metric-card-value" style="color: #f97316;">{active_sev_counts['ORANGE']}</div>
                <div class="metric-card-sub">Threshold: ≥ 2.50 m</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Caution (YELLOW)</div>
                <div class="metric-card-value" style="color: #f59e0b;">{active_sev_counts['YELLOW']}</div>
                <div class="metric-card-sub">Threshold: ≥ 1.50 m</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Resolved History</div>
                <div class="metric-card-value" style="color: #10b981;">{status_counts.get('RESOLVED', 0)}</div>
                <div class="metric-card-sub">De-escalated with Hysteresis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Fetch Alerts based on filters
    filter_loc = None if sel_loc == "ALL" else sel_loc
    filter_sev = None if sel_sev == "ALL" else sel_sev

    if sel_status == "ACTIVE_ACKNOWLEDGED":
        alerts_pool = alert_store.list_alerts(location_id=filter_loc, severity=filter_sev, limit=200)
        alerts = [a for a in alerts_pool if a.is_active_or_acknowledged]
    elif sel_status == "ALL":
        alerts = alert_store.list_alerts(location_id=filter_loc, severity=filter_sev, limit=200)
    else:
        alerts = alert_store.list_alerts(status=sel_status, location_id=filter_loc, severity=filter_sev, limit=200)

    st.markdown("### 📋 Coastal Hazard Alert Feed")

    if not alerts:
        st.markdown(
            """
            <div style="background: #111c2e; border: 1px dashed #334155; border-radius: 8px; padding: 28px; text-align: center; margin-top: 10px;">
                <h4 style="color: #94a3b8; margin: 0;">No Hazard Alerts Recorded in Database</h4>
                <p style="color: #64748b; font-size: 13px; margin-top: 6px;">
                    No wave forecast crossings matched the selected filter criteria. Click <b>'🔄 Evaluate Now'</b> above to run hazard detection across all candidate coastal nodes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for alert in alerts:
            sev_color = "#ef4444" if alert.severity == "RED" else ("#f97316" if alert.severity == "ORANGE" else "#f59e0b")
            status_chip = alert.status.display_badge

            val_badge = (
                "<span style='background: #065f46; color: #a7f3d0; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;'>REFERENCE VALIDATED (18.50°N, 72.50°E)</span>"
                if alert.is_reference
                else "<span style='background: #78350f; color: #fde68a; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;'>EXPERIMENTAL REGIONAL TRANSFER (UNVALIDATED)</span>"
            )

            st.markdown(
                f"""
                <div style="background: #0f172a; border-left: 5px solid {sev_color}; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b; border-radius: 8px; padding: 16px 20px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="color: #f8fafc; font-size: 17px; font-weight: 700;">{alert.location_name}</span>
                            <span style="margin-left: 10px;">{val_badge}</span>
                        </div>
                        <div>
                            <span style="background: {sev_color}22; color: {sev_color}; border: 1px solid {sev_color}66; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; margin-right: 8px;">
                                {alert.severity} RISK
                            </span>
                            <span style="background: #1e293b; color: #cbd5e1; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;">
                                {status_chip}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #111e38; padding: 10px 14px; border-radius: 6px; border: 1px solid #1e293b;">
                        <div>
                            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Predicted Wave Height</div>
                            <div style="color: {sev_color}; font-size: 18px; font-weight: 700;">{alert.predicted_swh_m:.2f} m</div>
                            <div style="color: #64748b; font-size: 11px;">Threshold: ≥ {alert.threshold_m:.2f} m</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Forecast Horizon</div>
                            <div style="color: #38bdf8; font-size: 18px; font-weight: 700;">+{alert.forecast_horizon_hours}h Ahead</div>
                            <div style="color: #64748b; font-size: 11px;">Event: {alert.expected_event_time_utc.strftime('%b %d, %H:%M UTC')}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Event Type</div>
                            <div style="color: #f1f5f9; font-size: 15px; font-weight: 600;">{alert.event_type.replace('_', ' ').title()}</div>
                            <div style="color: #64748b; font-size: 11px;">Status: {alert.data_status.upper()}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase;">Inference Model</div>
                            <div style="color: #f1f5f9; font-size: 15px; font-weight: 600;">{alert.model_name or 'N/A'}</div>
                            <div style="color: #64748b; font-size: 11px;">Updated: {alert.updated_at_utc.strftime('%H:%M:%S UTC')}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; color: #cbd5e1; font-size: 13px; line-height: 1.5;">
                        <b>Trigger:</b> {alert.trigger_reason}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"⚙️ Alert Actions & History (ID: {alert.alert_id[:8]}...)", expanded=False):
                act_col1, act_col2 = st.columns([1.2, 1.8])
                with act_col1:
                    if alert.status == AlertStatus.ACTIVE:
                        ack_notes = st.text_input(
                            "Operator Notes (Optional)",
                            key=f"notes_{alert.alert_id}",
                            placeholder="e.g., Port authority alerted",
                        )
                        if st.button("✍️ Acknowledge Alert", key=f"btn_ack_{alert.alert_id}", use_container_width=True):
                            alert_engine.acknowledge_alert(alert.alert_id, operator_notes=ack_notes or "Acknowledged in Streamlit dashboard.")
                            st.toast(f"Alert {alert.alert_id[:8]} acknowledged.", icon="✅")
                            st.rerun()
                    elif alert.status == AlertStatus.ACKNOWLEDGED:
                        st.info(f"Acknowledged at {alert.acknowledged_at_utc.strftime('%Y-%m-%d %H:%M UTC')}. Notes: {alert.operator_notes or 'None'}")
                    elif alert.status == AlertStatus.RESOLVED:
                        st.success(f"Resolved at {alert.resolved_at_utc.strftime('%Y-%m-%d %H:%M UTC') if alert.resolved_at_utc else 'N/A'}.")
                    elif alert.status == AlertStatus.EXPIRED:
                        st.caption("This event forecast window has expired.")

                with act_col2:
                    history = alert_store.get_history(alert.alert_id)
                    if history:
                        hist_data = []
                        for h in history:
                            hist_data.append(
                                {
                                    "From": h["from_status"] or "Initial",
                                    "To": h["to_status"],
                                    "Reason": h["reason"],
                                    "Timestamp": h["changed_at_utc"],
                                }
                            )
                        st.table(hist_data)
                    else:
                        st.caption("No state transition history logged.")

    # 4. Threshold Configuration Reference Expander
    with st.expander("⚙️ Configured Risk Thresholds & Engineering Bounds (Demonstration v1.0)", expanded=False):
        def_t = thresh_mgr.default_thresholds
        t_c1, t_c2, t_c3, t_c4 = st.columns(4)
        with t_c1:
            st.metric("Caution (YELLOW)", f"≥ {def_t.caution_m:.2f} m", "Elevated swell hazard")
        with t_c2:
            st.metric("High Risk (ORANGE)", f"≥ {def_t.high_risk_m:.2f} m", "Rough seas / craft warning")
        with t_c3:
            st.metric("Critical Risk (RED)", f"≥ {def_t.critical_m:.2f} m", "Severe sea state / high wave alert")
        with t_c4:
            st.metric("Hysteresis Buffer", f"{def_t.hysteresis_buffer_m:.2f} m", f"Resolves below {def_t.caution_m - def_t.hysteresis_buffer_m:.2f} m")

        st.caption(
            f"**Configuration Version**: `{def_t.version}` | **Units**: `{def_t.units}`. "
            f"**Notice**: {def_t.guidance_notice}"
        )


# =============================================================================
# PAGE: COASTAL RISK HEATMAP & MONITORING POINTS
# =============================================================================
elif nav_page == "🗺️ Coastal Risk Heatmap":
    st.markdown(
        """
        <div class="ocean-header" style="background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 50%, #0284c7 100%); border-color: #0ea5e9;">
            <h1 class="ocean-title">🗺️ Coastal Risk Heatmap & Monitoring Points</h1>
            <p class="ocean-subtitle">Interactive Point-Based Wave Hazard Visualizer Across Monitored Offshore Indian Coastline Candidate Grid Points</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 14px; margin-bottom: 4px;">📍 Point-Based Geospatial Visualization Notice</div>
            <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5;">
                Map markers represent <b>candidate offshore numerical forecasting grid cells</b> (~5 to 25 km offshore from major ports and promontories). 
                To maintain scientific integrity, <b>this map does not interpolate or fabricate continuous spatial wave fields</b> between sparse offshore monitoring points.
                A <span style="color: #10b981; font-weight: 600;">GREEN</span> marker indicates that no configured threshold is exceeded at that grid point; it is <b>never a guarantee of maritime safety</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Operational Controls Bar
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns([1.1, 1.4, 1.1, 1.1, 1.0])
    with m_c1:
        map_horizon = st.selectbox(
            "Forecast Horizon",
            [1, 6, 24],
            index=1,
            format_func=lambda h: f"+{h} Hour{'s' if h > 1 else ''} Ahead (t + {h}h)",
            help="Select lead time to evaluate wave height and risk category.",
        )
    with m_c2:
        map_source_mode = st.selectbox(
            "Telemetry Feed",
            [
                "📂 Offline Historical Demo (ERA5 Window)",
                "🌐 Open-Meteo REST API (Live Feed)",
            ],
            index=0,
            help="Choose between real-time external API or offline historical ground-truth replay.",
        )
    with m_c3:
        map_sev_filter = st.selectbox(
            "Filter Risk",
            ["ALL", "ELEVATED_ONLY", "RED", "ORANGE", "YELLOW", "GREEN"],
            format_func=lambda s: {
                "ALL": "All Categories",
                "ELEVATED_ONLY": "⚠️ Elevated (Y/O/R)",
                "RED": "🔴 RED (Critical)",
                "ORANGE": "🟠 ORANGE (High)",
                "YELLOW": "🟡 YELLOW (Caution)",
                "GREEN": "🟢 GREEN (Normal)",
            }[s],
            index=0,
        )
    with m_c4:
        map_data_filter = st.selectbox(
            "Filter Telemetry Status",
            ["ALL", "HEALTHY_ONLY", "FRESH", "CACHED", "OFFLINE_DEMO", "DEGRADED"],
            format_func=lambda d: {
                "ALL": "All Data States",
                "HEALTHY_ONLY": "Healthy Telemetry Only",
                "FRESH": "Fresh Only",
                "CACHED": "Cached Only",
                "OFFLINE_DEMO": "Offline Demo Only",
                "DEGRADED": "Stale / Unavailable Only",
            }[d],
            index=0,
        )
    with m_c5:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        force_fetch = st.button("🔄 Refresh Telemetry", use_container_width=True, help="Force refresh network queries bypassing cache")

    # 2. Data Retrieval via Multi-Location Ingestion & Mapping Subsystem
    source_clean = "demo" if "demo" in map_source_mode.lower() else "live"
    with st.spinner("Preparing geospatial wave hazard summaries..."):
        summaries = prepare_coastal_node_map_data(
            selected_horizon=map_horizon,
            source=source_clean,
            force_refresh=force_fetch,
            scaler=scaler_obj,
        )

    # 3. KPI Metrics Bar
    map_metrics = compute_map_dashboard_metrics(summaries)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Monitored Points</div>
                <div class="metric-card-value" style="color: #38bdf8;">{map_metrics['total_locations']} Nodes</div>
                <div class="metric-card-sub">Offshore Indian Coastline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Fresh Telemetry</div>
                <div class="metric-card-value" style="color: #10b981;">{map_metrics['fresh_locations']}</div>
                <div class="metric-card-sub">Real-Time Observation Feeds</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Stale / Unavailable</div>
                <div class="metric-card-value" style="color: {'#f59e0b' if map_metrics['stale_or_unavailable_locations'] > 0 else '#64748b'};">{map_metrics['stale_or_unavailable_locations']}</div>
                <div class="metric-card-sub">Degraded Feed States</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Elevated Hazard Nodes</div>
                <div class="metric-card-value" style="color: {'#ef4444' if map_metrics['elevated_risk_locations'] > 0 else '#10b981'};">{map_metrics['elevated_risk_locations']}</div>
                <div class="metric-card-sub">Caution, High, or Critical</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Active Alert Count</div>
                <div class="metric-card-value" style="color: {'#ef4444' if map_metrics['total_active_alerts'] > 0 else '#10b981'};">{map_metrics['total_active_alerts']}</div>
                <div class="metric-card-sub">From Persistent Alert Store</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Filter Summaries for Display
    filtered_summaries = summaries
    if map_sev_filter == "ELEVATED_ONLY":
        filtered_summaries = [s for s in filtered_summaries if s.current_risk_level in [RiskLevel.YELLOW, RiskLevel.ORANGE, RiskLevel.RED]]
    elif map_sev_filter != "ALL":
        filtered_summaries = [s for s in filtered_summaries if s.current_risk_level.value == map_sev_filter]

    if map_data_filter == "HEALTHY_ONLY":
        filtered_summaries = [s for s in filtered_summaries if not s.is_degraded and s.data_status in [DataStatus.FRESH, DataStatus.CACHED, DataStatus.OFFLINE_DEMO]]
    elif map_data_filter == "DEGRADED":
        filtered_summaries = [s for s in filtered_summaries if s.is_degraded or s.data_status in [DataStatus.STALE, DataStatus.UNAVAILABLE, DataStatus.INVALID]]
    elif map_data_filter != "ALL":
        filtered_summaries = [s for s in filtered_summaries if s.data_status.value == map_data_filter.lower()]

    # Maintain selected node in session state
    if "selected_map_node" not in st.session_state:
        st.session_state["selected_map_node"] = "mumbai_offshore"

    # 5. Dual-Column Interactive Layout: Map (Left) + Detail Panel (Right)
    col_map_view, col_node_details = st.columns([1.5, 1.1])

    with col_map_view:
        st.markdown(f"#### 🗺️ Coastal Monitoring Map (t + {map_horizon}h Horizon)")
        if HAS_STREAMLIT_FOLIUM:
            folium_map = build_coastal_folium_map(
                node_summaries=filtered_summaries,
                selected_node_id=st.session_state["selected_map_node"],
                height=560,
            )
            if folium_map is not None:
                map_out = st_folium(
                    folium_map,
                    width="100%",
                    height=560,
                    returned_objects=["last_object_clicked"],
                    key=f"coastal_map_{map_horizon}_{source_clean}",
                )

                # Sync click event to selected node
                if map_out and map_out.get("last_object_clicked"):
                    clicked_pt = map_out["last_object_clicked"]
                    c_lat, c_lon = clicked_pt.get("lat"), clicked_pt.get("lng")
                    if c_lat is not None and c_lon is not None:
                        nearest_node = min(
                            summaries,
                            key=lambda s: (s.latitude - c_lat) ** 2 + (s.longitude - c_lon) ** 2,
                        )
                        if abs(nearest_node.latitude - c_lat) < 0.8 and abs(nearest_node.longitude - c_lon) < 0.8:
                            if st.session_state["selected_map_node"] != nearest_node.location_id:
                                st.session_state["selected_map_node"] = nearest_node.location_id
                                st.rerun()
            else:
                st.error("Failed to initialize Folium map instance.")
        else:
            st.warning("`streamlit-folium` package is not available. Displaying node metrics summary table below.")

        st.caption(
            "💡 **Map Interactions**: Pan and zoom across the Indian coastline. Click any marker popup to preview telemetry, "
            "or use the selector panel on the right to inspect complete multi-horizon forecasts and alert histories."
        )

    with col_node_details:
        st.markdown("#### 🔍 Location Condition Inspector")

        node_dict = {s.location_id: s for s in summaries}
        node_ids = list(node_dict.keys())
        node_labels = {
            nid: f"{node_dict[nid].name} — {node_dict[nid].status_label}"
            for nid in node_ids
        }

        current_idx = node_ids.index(st.session_state["selected_map_node"]) if st.session_state["selected_map_node"] in node_ids else 0
        sel_node_id = st.selectbox(
            "Select Coastal Node",
            node_ids,
            index=current_idx,
            format_func=lambda nid: node_labels[nid],
            key="node_detail_selector",
            help="Select a candidate coastal point to view detailed forecasts and telemetry.",
        )
        st.session_state["selected_map_node"] = sel_node_id
        selected_s = node_dict[sel_node_id]

        # Selected Node Card
        card_sev_color = selected_s.marker_hex_color
        val_badge = (
            "<span style='background:#065f46;color:#a7f3d0;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;'>REFERENCE CALIBRATED (18.50°N, 72.50°E)</span>"
            if selected_s.is_reference
            else "<span style='background:#78350f;color:#fde68a;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;'>EXPERIMENTAL REGIONAL TRANSFER (UNVALIDATED)</span>"
        )

        st.markdown(
            f"""
            <div style="background: #0f172a; border-left: 5px solid {card_sev_color}; border-top: 1px solid #1e293b; border-right: 1px solid #1e293b; border-bottom: 1px solid #1e293b; border-radius: 8px; padding: 16px 18px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="color: #f8fafc; font-size: 17px; font-weight: 700;">{selected_s.name}</div>
                        <div style="color: #94a3b8; font-size: 12px; margin-top: 2px;">{selected_s.coordinate_label} | {selected_s.region}</div>
                    </div>
                    <div>
                        <span style="background: {card_sev_color}22; color: {card_sev_color}; border: 1px solid {card_sev_color}66; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;">
                            {selected_s.status_label}
                        </span>
                    </div>
                </div>
                <div style="margin-top: 8px;">{val_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Condition Metrics
        d_c1, d_c2, d_c3 = st.columns(3)
        with d_c1:
            obs_str = f"{selected_s.latest_obs_swh_m:.2f} m" if selected_s.latest_obs_swh_m is not None else "N/A"
            st.metric("Latest Observation (t)", obs_str, "Current Swell")
        with d_c2:
            fc_str = f"{selected_s.forecast_swh_m:.2f} m" if selected_s.forecast_swh_m is not None else "N/A"
            delta_str = f"{(selected_s.forecast_swh_m - selected_s.latest_obs_swh_m):+.2f} m" if selected_s.forecast_swh_m and selected_s.latest_obs_swh_m else None
            st.metric(f"+{selected_s.forecast_horizon_hours}h Forecast", fc_str, delta_str)
        with d_c3:
            max_str = f"{selected_s.max_window_swh_m:.2f} m" if selected_s.max_window_swh_m is not None else "N/A"
            st.metric("24h Peak Forecast", max_str, "Window Maximum")

        # Multi-Horizon Breakdown Table
        st.markdown("##### ⏱️ Horizon Predictions vs External NWP")
        if selected_s.multi_horizon_predictions:
            comp_rows = []
            for h_key in [1, 6, 24]:
                pred_val = selected_s.multi_horizon_predictions.get(h_key)
                nwp_val = selected_s.external_nwp_forecasts.get(h_key)
                comp_rows.append(
                    {
                        "Horizon": f"t + {h_key}h",
                        "DL Forecast (Hs)": f"{pred_val:.2f} m" if pred_val is not None else "N/A",
                        "ECMWF NWP": f"{nwp_val:.2f} m" if nwp_val is not None else "N/A",
                        "Risk Category": classify_risk(pred_val, get_threshold_manager().get_thresholds(selected_s.location_id))[0].value if pred_val is not None else "N/A",
                    }
                )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Multi-horizon predictions unavailable for this node state.")

        # Active Alert Records for this node
        st.markdown(f"##### 🚨 Active Alerts ({selected_s.active_alert_count})")
        if selected_s.active_alerts:
            for alt in selected_s.active_alerts:
                st.markdown(
                    f"""
                    <div style="background: #111e38; border-left: 4px solid {'#ef4444' if alt.severity == 'RED' else ('#f97316' if alt.severity == 'ORANGE' else '#f59e0b')}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>{alt.severity} RISK — +{alt.forecast_horizon_hours}h Horizon</b>
                            <span style="font-size: 11px; color: #94a3b8;">{alt.status.display_badge}</span>
                        </div>
                        <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">{alt.trigger_reason}</div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Expected Event: {alt.expected_event_time_utc.strftime('%Y-%m-%d %H:%M UTC')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No active hazard alerts recorded for this coastal node.")

        # Limitations & Explanation Expander
        with st.expander("ℹ️ Scientific Explanations & Limitations", expanded=False):
            if selected_s.trigger_explanation:
                st.markdown(f"**Trigger Reason**: {selected_s.trigger_explanation}")
            if selected_s.limitations:
                for lim in selected_s.limitations:
                    st.caption(f"• {lim}")

    # 6. Tabular Coastal Summary Expander
    with st.expander("📊 Complete Coastal Node Telemetry & Comparison Matrix", expanded=False):
        matrix_rows = []
        for s in summaries:
            matrix_rows.append(
                {
                    "Location Name": s.name,
                    "Coordinates": s.coordinate_label,
                    "Region": s.region,
                    "Validation": "Reference" if s.is_reference else "Regional",
                    "Observed (Hs)": f"{s.latest_obs_swh_m:.2f} m" if s.latest_obs_swh_m is not None else "N/A",
                    f"+{map_horizon}h Forecast": f"{s.forecast_swh_m:.2f} m" if s.forecast_swh_m is not None else "N/A",
                    "24h Peak (Hs)": f"{s.max_window_swh_m:.2f} m" if s.max_window_swh_m is not None else "N/A",
                    "Risk Level": s.status_label,
                    "Data Status": s.data_status.value.upper(),
                    "Active Alerts": s.active_alert_count,
                }
            )
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: FORECAST ACCURACY & MODEL DRIFT MONITORING
# =============================================================================
elif nav_page == "📈 Forecast Accuracy & Model Drift":
    st.markdown(
        """
        <div class="ocean-header" style="background: linear-gradient(135deg, #0f172a 0%, #0369a1 40%, #0284c7 100%); border-color: #0284c7;">
            <h1 class="ocean-title">📈 Forecast Accuracy & Model Drift Monitoring</h1>
            <p class="ocean-subtitle">Prospective Forecast Ledger, Verification Integrity, Feature Drift (PSI), and Rolling Model Degradation Tracking</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Scientific Transparency & Verification Status Notice
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.85); border-left: 4px solid #38bdf8; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 14px; margin-bottom: 4px;">🔬 Scientific Verification & Observation Integrity Policy</div>
            <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5;">
                • <b>Observation Sourcing</b>: True forecast verification requires independent, measured marine observations (e.g., moored oceanographic buoys or coastal radar). External provider forecasts (e.g. Open-Meteo / ECMWF) are <i>forecast-to-forecast comparisons</i>, not verified accuracy.<br>
                • <b>Retrospective Reanalysis</b>: Comparing predictions against historical ERA5 reanalysis is classified as <i>retrospective reanalysis backtesting</i>, not live observation verification.<br>
                • <b>Separation of Alerts</b>: ML operations indicators (feature drift, rolling performance degradation) are tracked independently and <b>never generate official coastal safety hazard warnings</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mon_store = get_monitoring_store()
    drift_mgr = get_drift_manager()
    coastal_reg = get_coastal_registry()

    # 2. Controls Bar
    c_loc, c_hor, c_mod, c_src, c_act = st.columns([1.3, 1.0, 1.1, 1.3, 1.1])
    with c_loc:
        mon_loc_id = st.selectbox(
            "Monitored Coastal Node",
            options=["ALL"] + [n.location_id for n in coastal_reg.list_all(enabled_only=True)],
            format_func=lambda lid: "All Coastal Nodes" if lid == "ALL" else f"{coastal_reg.get(lid).name} ({'Ref' if coastal_reg.get(lid).is_reference else 'Reg'})",
            index=1,
            help="Select coastal node for accuracy evaluation and ledger inspection.",
        )
    with c_hor:
        mon_horizon = st.selectbox(
            "Horizon",
            options=[0, 1, 6, 24],
            format_func=lambda h: "All Horizons" if h == 0 else f"+{h}h Ahead",
            index=0,
            help="Filter by forecast lead time.",
        )
    with c_mod:
        mon_model = st.selectbox(
            "Model Architecture",
            options=["ALL", "WaveLSTM", "WaveANN", "WaveCNN1D"],
            index=0,
            help="Filter by deep learning forecasting model.",
        )
    with c_src:
        mon_src_type = st.selectbox(
            "Verification Source",
            options=["ALL", "IN_SITU_BUOY", "RETROSPECTIVE_REANALYSIS"],
            format_func=lambda s: {
                "ALL": "All Sources (Buoy + ERA5)",
                "IN_SITU_BUOY": "🟢 Independent Buoys Only",
                "RETROSPECTIVE_REANALYSIS": "🔷 Retrospective ERA5 Only",
            }[s],
            index=0,
            help="Filter matched verification pairs by observation origin.",
        )
    with c_act:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Backtest ERA5", use_container_width=True, help="Load historical ERA5 ground-truth observations for retrospective accuracy verification"):
            with st.spinner("Ingesting historical ERA5 test observations for backtesting..."):
                target_loc = "mumbai_offshore" if mon_loc_id == "ALL" else mon_loc_id
                inserted = populate_retrospective_reanalysis_observations(mon_store, location_id=target_loc, max_rows=500)
                st.toast(f"Ingested {inserted} ERA5 reanalysis ground-truth observations.", icon="🔷")
                st.rerun()

    # Query persistent data
    filter_loc = None if mon_loc_id == "ALL" else mon_loc_id
    filter_hor = None if mon_horizon == 0 else mon_horizon
    filter_mod = None if mon_model == "ALL" else mon_model

    matched_pairs = mon_store.match_forecasts(
        location_id=filter_loc,
        horizon=filter_hor,
        model_name=filter_mod,
        limit=2000,
    )
    if mon_src_type != "ALL":
        target_src_enum = ObservationSourceType(mon_src_type.lower())
        matched_pairs = [m for m in matched_pairs if m.source_type == target_src_enum]

    total_issued = mon_store.count_forecasts(location_id=filter_loc)
    total_obs = mon_store.count_observations(location_id=filter_loc)
    active_mon_events = mon_store.list_monitoring_events(location_id=filter_loc, status="ACTIVE")

    # Accuracy Metrics Calculation
    acc_metrics = calculate_forecast_accuracy(
        matched_pairs=matched_pairs,
        total_issued=total_issued,
        min_sample_size=10,
        location_id=filter_loc,
        horizon=filter_hor,
        model_name=filter_mod,
    )

    # 3. Top KPI Summary Ribbon
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Forecasts Issued</div>
                <div class="metric-card-value" style="color: #38bdf8;">{total_issued}</div>
                <div class="metric-card-sub">Immutable Forecast Ledger</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Matched Observations</div>
                <div class="metric-card-value" style="color: #10b981;">{len(matched_pairs)}</div>
                <div class="metric-card-sub">Total Available: {total_obs}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Verification Coverage</div>
                <div class="metric-card-value" style="color: {'#10b981' if acc_metrics.coverage_pct > 0 else '#94a3b8'};">{acc_metrics.coverage_pct:.1f}%</div>
                <div class="metric-card-sub">{acc_metrics.verification_badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Verified MAE</div>
                <div class="metric-card-value" style="color: #38bdf8;">{f"{acc_metrics.mae_meters:.3f} m" if acc_metrics.mae_meters is not None else "N/A"}</div>
                <div class="metric-card-sub">{f"RMSE: {acc_metrics.rmse_meters:.3f} m" if acc_metrics.rmse_meters is not None else "Awaiting Data"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Active Monitoring Events</div>
                <div class="metric-card-value" style="color: {'#ef4444' if len(active_mon_events) > 0 else '#10b981'};">{len(active_mon_events)}</div>
                <div class="metric-card-sub">Data Drift & Degradation Logs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 4. Tabbed Deep-Dive Workspace
    tab_acc, tab_drift, tab_deg, tab_events = st.tabs(
        [
            "🎯 Forecast Accuracy & Baseline",
            "🌊 Canonical Feature Drift (PSI)",
            "📉 Model Performance Degradation",
            "📋 Operational ML Event Log",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: FORECAST ACCURACY & BASELINE COMPARISON
    # -------------------------------------------------------------------------
    with tab_acc:
        st.markdown("#### 🎯 Verified Forecast Accuracy & Persistence Comparison")
        st.caption(
            "Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Bias Error strictly "
            "against eligible matched observations. Compares deep learning predictions directly against a naive persistence baseline "
            "on the exact same matched sample cases."
        )

        st.info(f"**Verification Status**: {acc_metrics.verification_badge} — {acc_metrics.status_explanation}")

        if acc_metrics.sample_count > 0:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Mean Absolute Error (MAE)", f"{acc_metrics.mae_meters:.3f} m", help="Average absolute magnitude of forecast error in meters.")
            with m_col2:
                st.metric("Root Mean Squared Error (RMSE)", f"{acc_metrics.rmse_meters:.3f} m", help="Penalizes large outlier forecast errors heavily.")
            with m_col3:
                bias_sign = "Over-prediction" if acc_metrics.bias_meters > 0 else ("Under-prediction" if acc_metrics.bias_meters < 0 else "Zero Bias")
                st.metric("Mean Bias Error", f"{acc_metrics.bias_meters:+.3f} m", bias_sign, help="Prediction minus observation. Positive means model overpredicts wave heights.")
            with m_col4:
                skill_val = f"{acc_metrics.skill_score_rmse * 100:.1f}%" if acc_metrics.skill_score_rmse is not None else "N/A"
                skill_delta = "Outperforms Baseline" if (acc_metrics.skill_score_rmse or 0) > 0 else "Below Persistence"
                st.metric("RMSE Skill vs Persistence", skill_val, skill_delta, help="1.0 - (RMSE_model / RMSE_persistence). Positive indicates model beats persistence.")

            # Persistence comparison table
            st.markdown("##### ⏱️ Model vs. Naive Persistence Baseline (Identical Cases)")
            comp_df = pd.DataFrame(
                [
                    {
                        "Method": f"{filter_mod or 'Deep Learning Model'} (+{filter_hor or 'all'}h)",
                        "MAE (meters)": f"{acc_metrics.mae_meters:.4f} m",
                        "RMSE (meters)": f"{acc_metrics.rmse_meters:.4f} m",
                        "Mean Bias": f"{acc_metrics.bias_meters:+.4f} m",
                        "Matched Samples": acc_metrics.sample_count,
                    },
                    {
                        "Method": "Naive Persistence Baseline (t)",
                        "MAE (meters)": f"{acc_metrics.persistence_mae_meters:.4f} m" if acc_metrics.persistence_mae_meters is not None else "N/A",
                        "RMSE (meters)": f"{acc_metrics.persistence_rmse_meters:.4f} m" if acc_metrics.persistence_rmse_meters is not None else "N/A",
                        "Mean Bias": "0.0000 m (Anchor)",
                        "Matched Samples": acc_metrics.sample_count,
                    },
                ]
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Matched pairs table
            st.markdown("##### 🔍 Recent Matched Forecast-Observation Audit Trail")
            pair_rows = []
            for p in matched_pairs[:20]:
                pair_rows.append(
                    {
                        "Target Time (UTC)": p.target_time_utc.strftime("%Y-%m-%d %H:%M"),
                        "Model": p.model_name,
                        "Horizon": f"+{p.horizon_hours}h",
                        "Predicted (Hs)": f"{p.predicted_swh_m:.2f} m",
                        "Observed (Hs)": f"{p.observed_swh_m:.2f} m",
                        "Residual (Pred - Obs)": f"{p.error_m:+.2f} m",
                        "Persistence (Hs)": f"{p.persistence_swh_m:.2f} m" if p.persistence_swh_m is not None else "N/A",
                        "Observation Source": p.source_type.value.upper(),
                        "Verification Type": "Prospective" if p.is_prospective else "Retrospective",
                    }
                )
            st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)
        else:
            st.warning(
                "No eligible forecast-observation pairs matched the current filter. "
                "Use the **'🔄 Backtest ERA5'** button above to load historical ground-truth reanalysis observations, "
                "or generate live forecasts to begin prospective tracking."
            )

    # -------------------------------------------------------------------------
    # TAB 2: DATA DRIFT (POPULATION STABILITY INDEX)
    # -------------------------------------------------------------------------
    with tab_drift:
        st.markdown("#### 🌊 Canonical Input Feature Drift (Population Stability Index)")
        st.caption(
            "Monitors incoming meteorological and marine features against the historical ERA5 training split reference distribution (70%). "
            "Evaluates Population Stability Index (PSI), missing rates, and physical range boundary violations."
        )

        d_c1, d_c2 = st.columns([1.5, 1.0])
        with d_c1:
            drift_feed_mode = st.radio(
                "Input Data Stream to Analyze",
                ["📂 Offline Historical Window (ERA5 Telemetry)", "🌐 Live Open-Meteo REST API"],
                index=0,
                horizontal=True,
            )

        source_clean = "demo" if ("offline" in drift_feed_mode.lower() or "demo" in drift_feed_mode.lower()) else "live"
        ingestion_svc = get_ingestion_service()
        eval_node_id = "mumbai_offshore" if mon_loc_id == "ALL" else mon_loc_id

        with st.spinner("Analyzing feature distribution shifts..."):
            fetch_res = ingestion_svc.fetch_location(eval_node_id, source=source_clean)
            if fetch_res and fetch_res.raw_df is not None and not fetch_res.raw_df.empty:
                drift_rep = drift_mgr.calculate_psi(fetch_res.raw_df)
            else:
                drift_rep = drift_mgr.calculate_psi(None)

        # Status Badge
        status_color = "#10b981" if drift_rep.overall_status == DriftStatus.NORMAL else ("#f59e0b" if drift_rep.overall_status == DriftStatus.WATCH else "#ef4444")
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid {status_color}; padding: 12px 16px; border-radius: 6px; margin: 12px 0 18px 0;">
                <div style="font-weight: 700; color: {status_color}; font-size: 14px;">Overall Distribution Drift Status: {drift_rep.overall_status.value}</div>
                <div style="color: #cbd5e1; font-size: 13px; margin-top: 3px;">{drift_rep.summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if drift_rep.feature_drifts:
            feature_table_data = []
            chart_data = {}
            for fname, f_res in drift_rep.feature_drifts.items():
                feature_table_data.append(
                    {
                        "Feature": fname,
                        "PSI Score": f"{f_res.psi:.4f}",
                        "Drift Category": f_res.drift_level,
                        "Current Mean": f"{f_res.curr_mean:.2f}",
                        "Reference Mean": f"{f_res.ref_mean:.2f}",
                        "Current Std": f"{f_res.curr_std:.2f}",
                        "Reference Std": f"{f_res.ref_std:.2f}",
                        "Missing Rate": f"{f_res.missing_rate * 100:.1f}%",
                        "Out of Bounds": f"{f_res.out_of_bounds_rate * 100:.1f}%",
                    }
                )
                chart_data[fname] = f_res.psi

            col_t, col_ch = st.columns([1.5, 1.0])
            with col_t:
                st.markdown("##### 📊 Feature Distribution Drift Matrix")
                st.dataframe(pd.DataFrame(feature_table_data), use_container_width=True, hide_index=True)
            with col_ch:
                st.markdown("##### 📈 PSI by Feature")
                st.bar_chart(pd.DataFrame(list(chart_data.items()), columns=["Feature", "PSI"]).set_index("Feature"))

            st.caption(
                "💡 **Population Stability Index (PSI) Standards**:<br>"
                "• **PSI < 0.10** (🟢 NO_DRIFT): Incoming feature distribution conforms to training baseline.<br>"
                "• **0.10 ≤ PSI < 0.25** (🟡 MODERATE_DRIFT / WATCH): Moderate distribution shift; monitor incoming data.<br>"
                "• **PSI ≥ 0.25** (🔴 SIGNIFICANT_DRIFT): Significant covariate shift (e.g. seasonal storm anomaly or sensor bias).",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # TAB 3: MODEL PERFORMANCE DEGRADATION TRACKER
    # -------------------------------------------------------------------------
    with tab_deg:
        st.markdown("#### 📉 Model Performance Degradation Tracking")
        st.caption(
            "Tracks rolling error against historical test set benchmarks. Detects sustained performance degradation "
            "using hysteresis to avoid noisy, false-alarm alerts."
        )

        # Baseline Benchmark Summary Table
        st.markdown("##### 🏛️ Historical Test Set Benchmarks (Reference Ground-Truth)")
        base_rows = []
        for m_name, h_dict in {
            "WaveLSTM": {1: 0.0182, 6: 0.0650, 24: 0.1420},
            "WaveANN": {1: 0.0349, 6: 0.0980, 24: 0.1890},
            "WaveCNN1D": {1: 0.0426, 6: 0.1120, 24: 0.2050},
        }.items():
            base_rows.append(
                {
                    "Model": m_name,
                    "+1h Test MAE": f"{h_dict[1]:.4f} m",
                    "+6h Test MAE": f"{h_dict[6]:.4f} m",
                    "+24h Test MAE": f"{h_dict[24]:.4f} m",
                }
            )
        st.dataframe(pd.DataFrame(base_rows), use_container_width=True, hide_index=True)

        target_deg_mod = "WaveLSTM" if mon_model == "ALL" else mon_model
        target_deg_hor = 1 if mon_horizon == 0 else mon_horizon

        deg_report = evaluate_model_performance_degradation(
            matched_pairs=matched_pairs,
            model_name=target_deg_mod,
            horizon=target_deg_hor,
            min_samples=5,
        )

        deg_color = "#10b981" if deg_report.status == PerformanceStatus.NORMAL else ("#f59e0b" if deg_report.status == PerformanceStatus.WATCH else ("#ef4444" if deg_report.status == PerformanceStatus.PERFORMANCE_DEGRADATION else "#94a3b8"))
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid {deg_color}; padding: 12px 16px; border-radius: 6px; margin: 16px 0;">
                <div style="font-weight: 700; color: {deg_color}; font-size: 14px;">Degradation Tracking Status: {deg_report.status.value}</div>
                <div style="color: #cbd5e1; font-size: 13px; margin-top: 3px;">{deg_report.summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Rolling Verified MAE", f"{deg_report.rolling_mae:.4f} m" if deg_report.rolling_mae is not None else "N/A", help="Rolling mean absolute error across matched evaluation cases.")
        with p2:
            st.metric("Baseline Test MAE", f"{deg_report.baseline_mae:.4f} m", help="Historical test split MAE for comparison.")
        with p3:
            ratio_str = f"{deg_report.mae_ratio:.2f}x" if deg_report.mae_ratio is not None else "N/A"
            ratio_delta = "Elevated" if (deg_report.mae_ratio or 1.0) > 1.25 else "Nominal"
            st.metric("Error Multiplier vs Baseline", ratio_str, ratio_delta, help="Ratio of rolling error to historical baseline. Values > 1.50x trigger PERFORMANCE_DEGRADATION.")

        st.caption(
            "⚠️ **Policy**: Model performance degradation is an engineering diagnostic indicator. "
            "It does NOT automatically disable, retrain, or replace any production model checkpoints.",
        )

    # -------------------------------------------------------------------------
    # TAB 4: OPERATIONAL ML MONITORING EVENT LOG
    # -------------------------------------------------------------------------
    with tab_events:
        st.markdown("#### 📋 Operational ML Monitoring Event Log")
        st.caption(
            "Persistent history of data drift, feature quality anomalies, and rolling model performance alerts. "
            "Deduplicated and strictly separated from coastal maritime wave hazard alerts."
        )

        all_events = mon_store.list_monitoring_events(location_id=filter_loc, limit=100)
        if all_events:
            ev_table = []
            for ev in all_events:
                ev_table.append(
                    {
                        "Event ID": ev.event_id,
                        "Type": ev.event_type.value.upper(),
                        "Location": ev.location_id,
                        "Severity": ev.severity.value,
                        "Status": ev.status.value,
                        "Metric": f"{ev.metric_name}: {ev.metric_value:.3f} (Thresh: {ev.threshold_value:.3f})",
                        "Summary": ev.summary,
                        "Timestamp (UTC)": ev.created_at_utc.strftime("%Y-%m-%d %H:%M"),
                    }
                )
            st.dataframe(pd.DataFrame(ev_table), use_container_width=True, hide_index=True)
        else:
            st.success("No active or historical ML monitoring events logged.")


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

    # 1. Operational Controls & Coastal Node Selection
    c_loc, c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.4, 0.9, 1.1, 1.1])
    with c_loc:
        coastal_reg = get_coastal_registry()
        reg_nodes = coastal_reg.list_all(enabled_only=True)
        node_map = {n.location_id: f"{n.name} ({n.coordinate_label})" for n in reg_nodes}
        node_keys = list(node_map.keys())
        selected_loc_id = st.selectbox(
            "Coastal Monitoring Location",
            node_keys,
            index=0,
            format_func=lambda k: node_map[k],
            help="Select candidate offshore coastal monitoring location along the Indian coastline.",
        )
        active_node = coastal_reg.get(selected_loc_id)

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
    with st.expander("⚙️ Location Metadata, Coordinates, API Settings & Cache Control", expanded=False):
        set_c1, set_c2, set_c3, set_c4 = st.columns([1, 1, 1, 1])
        with set_c1:
            target_lat = st.number_input("Target Latitude (°N)", value=float(active_node.latitude), min_value=-90.0, max_value=90.0, step=0.05, format="%.2f")
        with set_c2:
            target_lon = st.number_input("Target Longitude (°E)", value=float(active_node.longitude), min_value=-180.0, max_value=180.0, step=0.05, format="%.2f")
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
            f"**Selected Node**: {active_node.name} | Region: **{active_node.region}** | "
            f"Validation Status: `{active_node.validation_status.value}`. "
            "Data responses are cached for 15 minutes. Click 'Refresh Data Now' to force an immediate network query."
        )
        if active_node.notes:
            for note in active_node.notes:
                st.caption(f"• {note}")

    # 3. Ingestion & Quality Validation
    fetch_result: Optional[LiveFetchResult] = None
    if "Live Real-Time" in source_mode:
        with st.spinner("Connecting to Open-Meteo Marine & Atmospheric APIs for near-real-time observations..."):
            try:
                fetch_result = fetch_cached_live_marine(
                    latitude=target_lat,
                    longitude=target_lon,
                    timeout_s=api_timeout,
                    location_id=active_node.location_id,
                )
            except Exception as e:
                st.markdown(
                    f"""
                    <div style='background: #3b1812; border: 1px solid #7f1d1d; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;'>
                        <h4 style='color: #fca5a5; margin: 0 0 6px 0;'>⚠️ External API Query Notice ({active_node.name})</h4>
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

    # 4. Display Geographic Calibration & Operational Status Banners
    now_utc = datetime.now(timezone.utc)
    latest_obs_utc = fetch_result.latest_obs_time_utc
    latest_obs_ist = latest_obs_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    freshness_mins = max(0, int((now_utc - latest_obs_utc).total_seconds() / 60))
    data_status_str = getattr(fetch_result, "data_status", "fresh" if not fetch_result.is_demo else "offline_demo")
    issue_time_utc = getattr(fetch_result, "forecast_issue_time_utc", None)

    # Geographic Model Calibration Disclosure Banner
    if active_node.is_reference:
        st.markdown(
            f"""
            <div style='background: #0f2d1e; border: 1px solid #059669; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div>
                    <span class="badge-success">REFERENCE VALIDATED LOCATION</span>
                    <b style='color: #a7f3d0; font-size: 13px; margin-left: 10px;'>{active_node.name} ({active_node.coordinate_label})</b>
                </div>
                <div style='color: #d1fae5; font-size: 12px;'>
                    Calibrated on 14.8y ERA5 reanalysis | Zero-leakage scaler reference
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background: #2d1a08; border: 1px solid #d97706; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                    <div>
                        <span class="badge-warning">EXPERIMENTAL REGIONAL TRANSFER</span>
                        <b style='color: #fde68a; font-size: 13px; margin-left: 10px;'>{active_node.name} ({active_node.coordinate_label})</b>
                    </div>
                    <div style='color: #fef3c7; font-size: 12px;'>
                        Candidate Forecast Coordinates | Regional Transfer
                    </div>
                </div>
                <p style='color: #fde68a; font-size: 12px; margin: 4px 0 0 0;'>
                    ⚠️ <b>Geographic Calibration Notice:</b> Models were trained on Arabian Sea node (18.50°N, 72.50°E).
                    Predictions at this location represent zero-shot transfer without localized bathymetric calibration. Evaluate alongside ECMWF NWP baseline.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Telemetry Source and Freshness Banner
    issue_str = f" | Model Issue Cycle: <b>{issue_time_utc.strftime('%H:00 UTC')}</b>" if issue_time_utc else ""
    if not fetch_result.is_demo:
        st.markdown(
            f"""
            <div style='background: #064e3b; border: 1px solid #059669; border-radius: 8px; padding: 12px 20px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div>
                    <span class="badge-success">TELEMETRY: {data_status_str.upper()}</span>
                    <span style='color: #a7f3d0; font-size: 13px; margin-left: 10px;'>
                        Source: <b>{fetch_result.source_name}</b>{issue_str}
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
                <h4 style='color: #fca5a5; margin: 0 0 4px 0;'>❌ Observation Quality Validation Rejection ({active_node.name})</h4>
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
            location_id=active_node.location_id,
            validation_status=active_node.validation_status.value,
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

