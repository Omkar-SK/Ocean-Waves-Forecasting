"""
Automated Unit and Integration Tests for Phase 2:
Multi-Location Live Data Foundation.

Validates:
1. Coastal Node Registry (coordinate bounds, uniqueness, reference status, enable/disable).
2. Marine Data Provider Abstraction (Open-Meteo REST parsing, UTC handling, error resilience).
3. Explicit Data Status Lifecycle (FRESH, CACHED, STALE, OFFLINE_DEMO, UNAVAILABLE, INVALID).
4. Multi-Location Ingestion Service (bounded concurrency, per-location error isolation, TTL caching).
5. Backward Compatibility (predict_live_forecast, FastAPI /api/forecast/locations and /api/forecast/live).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from backend.main import app
from src.data.dataset import TimeSeriesScaler
from src.data.live_service import (
    predict_live_forecast,
    validate_and_prepare_input_window,
)
from src.intelligence.ingestion import MultiLocationIngestionService
from src.intelligence.nodes import (
    CoastalNode,
    CoastalNodeRegistry,
    DEFAULT_COASTAL_NODES,
    ValidationStatus,
    create_default_registry,
)
from src.intelligence.provider import (
    BaseMarineDataProvider,
    DataStatus,
    OfflineDemoProvider,
    OpenMeteoProvider,
    ProviderFetchResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


@pytest.fixture
def sample_raw_dataframe():
    """Generates 48 continuous hourly rows of valid marine and weather observations."""
    base_time = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    times = [base_time + pd.Timedelta(hours=i) for i in range(48)]
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "valid_time": times,
            "swh": np.linspace(0.8, 1.2, 48),
            "mwp": np.linspace(7.5, 8.5, 48),
            "mwd": np.linspace(200.0, 240.0, 48),
            "wind_speed": np.linspace(4.0, 7.0, 48),
            "wind_dir": np.linspace(340.0, 360.0, 48),
        }
    )
    return df


@pytest.fixture
def trained_scaler():
    """Loads the pre-fitted training scaler."""
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    assert scaler_path.is_file(), "Scaler artifact missing."
    return TimeSeriesScaler.load(scaler_path)


# =============================================================================
# 1. Coastal Node Registry Tests
# =============================================================================

def test_node_creation_valid():
    """Verify that a valid CoastalNode initializes with all required fields."""
    node = CoastalNode(
        location_id="test_port",
        name="Test Port Node",
        latitude=15.5,
        longitude=73.8,
        region="Test Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="A test coastal monitoring node.",
    )
    assert node.location_id == "test_port"
    assert node.coordinate_label == "15.50°N, 73.80°E"
    assert node.is_reference is False
    assert node.validation_status == ValidationStatus.EXPERIMENTAL_UNVALIDATED

    d = node.to_dict()
    assert d["location_id"] == "test_port"
    assert d["latitude"] == 15.5
    assert d["longitude"] == 73.8
    assert d["coordinate_label"] == "15.50°N, 73.80°E"
    assert d["validation_status"] == "experimental_unvalidated"


def test_node_coordinate_validation():
    """Verify that out-of-bounds latitude and longitude are rejected."""
    # Latitude > 90
    with pytest.raises(ValueError, match="Latitude 95.0 is out of valid bounds"):
        CoastalNode(
            location_id="invalid_lat",
            name="Invalid Lat",
            latitude=95.0,
            longitude=72.0,
            region="Test",
        )

    # Latitude < -90
    with pytest.raises(ValueError, match="Latitude -91.0 is out of valid bounds"):
        CoastalNode(
            location_id="invalid_lat_neg",
            name="Invalid Lat Neg",
            latitude=-91.0,
            longitude=72.0,
            region="Test",
        )

    # Longitude > 180
    with pytest.raises(ValueError, match="Longitude 185.0 is out of valid bounds"):
        CoastalNode(
            location_id="invalid_lon",
            name="Invalid Lon",
            latitude=18.0,
            longitude=185.0,
            region="Test",
        )

    # Longitude < -180
    with pytest.raises(ValueError, match="Longitude -185.0 is out of valid bounds"):
        CoastalNode(
            location_id="invalid_lon_neg",
            name="Invalid Lon Neg",
            latitude=18.0,
            longitude=-185.0,
            region="Test",
        )


def test_node_location_id_validation():
    """Verify that location_id must be lowercase alphanumeric with underscores."""
    with pytest.raises(ValueError, match="Invalid location_id"):
        CoastalNode(
            location_id="Invalid-ID!",
            name="Bad ID",
            latitude=18.0,
            longitude=72.0,
            region="Test",
        )

    with pytest.raises(ValueError, match="location_id must be a non-empty string"):
        CoastalNode(
            location_id="",
            name="Empty ID",
            latitude=18.0,
            longitude=72.0,
            region="Test",
        )


def test_node_reference_status_consistency():
    """Verify that is_reference=True requires validation_status=REFERENCE_VALIDATED."""
    with pytest.raises(ValueError, match="Reference location must have validation_status = REFERENCE_VALIDATED"):
        CoastalNode(
            location_id="fake_reference",
            name="Fake Reference",
            latitude=18.0,
            longitude=72.0,
            region="Test",
            is_reference=True,
            validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        )


def test_registry_registration_and_duplicate_rejection():
    """Verify registering nodes and rejecting duplicates."""
    reg = CoastalNodeRegistry()
    node = CoastalNode(
        location_id="node_a",
        name="Node A",
        latitude=10.0,
        longitude=75.0,
        region="Test",
    )
    reg.register(node)
    assert len(reg) == 1
    assert "node_a" in reg
    assert reg.get("node_a").name == "Node A"

    with pytest.raises(ValueError, match="already registered"):
        reg.register(node)


def test_registry_enable_disable_toggle():
    """Verify toggling enabled status and protecting reference node."""
    reg = create_default_registry()
    assert reg.get("goa_mormugao").enabled is True

    # Disable Goa
    updated = reg.set_enabled("goa_mormugao", False)
    assert updated.enabled is False
    assert reg.get("goa_mormugao").enabled is False
    assert "goa_mormugao" not in [n.location_id for n in reg.list_all(enabled_only=True)]

    # Attempt to disable Mumbai reference node must fail
    with pytest.raises(ValueError, match="The primary reference location cannot be disabled"):
        reg.set_enabled("mumbai_offshore", False)

    # Re-enable Goa
    reg.set_enabled("goa_mormugao", True)
    assert reg.get("goa_mormugao").enabled is True


def test_default_registry_composition():
    """Verify default 8 coastal candidate nodes and reference integrity."""
    reg = create_default_registry()
    assert len(reg) == 8

    # Reference node check
    ref = reg.get_reference_node()
    assert ref.location_id == "mumbai_offshore"
    assert ref.is_reference is True
    assert ref.validation_status == ValidationStatus.REFERENCE_VALIDATED
    assert ref.latitude == 18.50
    assert ref.longitude == 72.50

    # Non-reference nodes check
    non_ref_ids = [
        "goa_mormugao",
        "mangalore_port",
        "kochi_coast",
        "kanyakumari",
        "chennai_port",
        "visakhapatnam_port",
        "paradip_port",
    ]
    for loc_id in non_ref_ids:
        node = reg.get(loc_id)
        assert node.is_reference is False
        assert node.validation_status == ValidationStatus.EXPERIMENTAL_UNVALIDATED


# =============================================================================
# 2. Marine Data Provider Abstraction Tests
# =============================================================================

def _build_mock_open_meteo_payload(num_hours=48):
    """Helper to generate realistic Open-Meteo JSON responses for testing."""
    base_time = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=24)
    times = [(base_time + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(num_hours)]

    marine_data = {
        "latitude": 18.5,
        "longitude": 72.5,
        "generationtime_ms": 1.25,
        "utc_offset_seconds": 0,
        "hourly": {
            "time": times,
            "wave_height": [round(0.8 + 0.01 * i, 2) for i in range(num_hours)],
            "wave_period": [round(7.5 + 0.02 * i, 2) for i in range(num_hours)],
            "wave_direction": [round(220.0 + 0.5 * i, 1) for i in range(num_hours)],
        },
    }

    weather_data = {
        "latitude": 18.5,
        "longitude": 72.5,
        "generationtime_ms": 0.85,
        "utc_offset_seconds": 0,
        "hourly": {
            "time": times,
            "wind_speed_10m": [round(5.0 + 0.05 * i, 2) for i in range(num_hours)],
            "wind_direction_10m": [round(340.0 + 0.2 * i, 1) for i in range(num_hours)],
        },
    }
    return marine_data, weather_data


def test_open_meteo_provider_success():
    """Verify OpenMeteoProvider parses responses and aligns UTC timestamps."""
    provider = OpenMeteoProvider(max_retries=0)
    node = DEFAULT_COASTAL_NODES[0]

    marine_mock, weather_mock = _build_mock_open_meteo_payload(48)

    def mock_get(url, headers, timeout_s):
        if "marine-api" in url:
            return marine_mock
        return weather_mock

    with patch.object(provider, "_execute_http_get", side_effect=mock_get):
        result = provider.fetch_telemetry(node)

    assert result.is_success is True
    assert result.data_status in [DataStatus.FRESH, DataStatus.STALE]
    assert result.location_id == "mumbai_offshore"
    assert result.raw_df is not None
    assert len(result.raw_df) == 48
    assert "swh" in result.raw_df.columns
    assert "wind_speed" in result.raw_df.columns
    assert result.raw_df["valid_time"].dt.tz is not None  # Must be timezone-aware UTC
    assert result.forecast_issue_time_utc is not None
    assert result.error_message is None


def test_open_meteo_provider_network_error():
    """Verify OpenMeteoProvider assigns UNAVAILABLE upon network failure."""
    provider = OpenMeteoProvider(max_retries=0)
    node = DEFAULT_COASTAL_NODES[0]

    with patch.object(provider, "_execute_http_get", side_effect=ConnectionError("HTTP 500: Gateway Timeout")):
        result = provider.fetch_telemetry(node)

    assert result.is_success is False
    assert result.data_status == DataStatus.UNAVAILABLE
    assert result.raw_df.empty
    assert "Gateway Timeout" in result.error_message


def test_open_meteo_provider_malformed_json():
    """Verify OpenMeteoProvider handles missing keys without unhandled exceptions."""
    provider = OpenMeteoProvider(max_retries=0)
    node = DEFAULT_COASTAL_NODES[0]

    with patch.object(provider, "_execute_http_get", return_value={"error": True, "reason": "Bad request"}):
        result = provider.fetch_telemetry(node)

    assert result.is_success is False
    assert result.data_status == DataStatus.UNAVAILABLE
    assert "Malformed response" in result.error_message


def test_offline_demo_provider():
    """Verify OfflineDemoProvider replaying ERA5 historical archive."""
    demo_provider = OfflineDemoProvider()
    node = DEFAULT_COASTAL_NODES[0]

    result = demo_provider.fetch_telemetry(node)

    assert result.is_success is True
    assert result.data_status == DataStatus.OFFLINE_DEMO
    assert result.is_demo is True
    assert len(result.raw_df) >= 24
    assert result.error_message is None
    # Must never be labeled as FRESH
    assert result.data_status != DataStatus.FRESH


# =============================================================================
# 3. Multi-Location Ingestion Service Tests
# =============================================================================

def test_ingestion_caching_and_hit():
    """Verify that IngestionService caches responses and serves CACHED state within TTL."""
    mock_provider = MagicMock(spec=BaseMarineDataProvider)
    mock_provider.provider_name = "MockLiveProvider"
    mock_provider.attribution = "Mock Data"

    fixed_retrieval_time = datetime(2026, 10, 27, 6, 0, 0, tzinfo=timezone.utc)
    raw_df = pd.DataFrame(
        {
            "valid_time": pd.date_range("2026-10-25", periods=48, freq="h", tz="UTC"),
            "swh": np.ones(48),
            "mwp": np.ones(48) * 8.0,
            "mwd": np.ones(48) * 220.0,
            "wind_speed": np.ones(48) * 5.0,
            "wind_dir": np.ones(48) * 340.0,
        }
    )

    mock_result = ProviderFetchResult(
        provider_name="MockLiveProvider",
        location_id="mumbai_offshore",
        latitude=18.5,
        longitude=72.5,
        retrieval_time_utc=fixed_retrieval_time,
        data_status=DataStatus.FRESH,
        raw_df=raw_df,
        forecast_issue_time_utc=fixed_retrieval_time,
        latest_obs_time_utc=fixed_retrieval_time,
    )
    mock_provider.fetch_telemetry.return_value = mock_result

    service = MultiLocationIngestionService(live_provider=mock_provider, cache_ttl_s=300)

    # 1st call: queries provider -> FRESH
    res1 = service.fetch_location("mumbai_offshore", source="live")
    assert res1.data_status == DataStatus.FRESH
    assert mock_provider.fetch_telemetry.call_count == 1

    # 2nd call: within TTL -> CACHED, preserving original retrieval_time_utc
    res2 = service.fetch_location("mumbai_offshore", source="live")
    assert res2.data_status == DataStatus.CACHED
    assert res2.retrieval_time_utc == fixed_retrieval_time
    assert mock_provider.fetch_telemetry.call_count == 1  # No additional provider call!

    # 3rd call: force_refresh=True -> bypasses cache, queries provider
    res3 = service.fetch_location("mumbai_offshore", source="live", force_refresh=True)
    assert res3.data_status == DataStatus.FRESH
    assert mock_provider.fetch_telemetry.call_count == 2


def test_ingestion_bounded_concurrency_and_error_isolation():
    """Verify concurrent ingestion with error isolation (one node failure doesn't crash batch)."""
    reg = CoastalNodeRegistry(
        [
            CoastalNode("node_success_1", "Node 1", 18.0, 72.0, "Reg1"),
            CoastalNode("node_failing", "Node Failing", 15.0, 73.0, "Reg2"),
            CoastalNode("node_success_2", "Node 2", 12.0, 74.0, "Reg3"),
        ]
    )

    raw_df = pd.DataFrame(
        {
            "valid_time": pd.date_range("2026-10-25", periods=24, freq="h", tz="UTC"),
            "swh": np.ones(24),
            "mwp": np.ones(24),
            "mwd": np.ones(24),
            "wind_speed": np.ones(24),
            "wind_dir": np.ones(24),
        }
    )

    class MockFlakyProvider(BaseMarineDataProvider):
        @property
        def provider_name(self) -> str:
            return "FlakyProvider"

        @property
        def attribution(self) -> str:
            return "Test"

        def fetch_telemetry(self, node, past_days=2, forecast_days=2, timeout_s=10):
            if node.location_id == "node_failing":
                raise ConnectionResetError("Simulated remote connection abort.")
            return ProviderFetchResult(
                provider_name="FlakyProvider",
                location_id=node.location_id,
                latitude=node.latitude,
                longitude=node.longitude,
                retrieval_time_utc=datetime.now(timezone.utc),
                data_status=DataStatus.FRESH,
                raw_df=raw_df,
            )

    service = MultiLocationIngestionService(
        registry=reg,
        live_provider=MockFlakyProvider(),
        max_workers=2,
    )

    results = service.fetch_all_locations(source="live")

    assert len(results) == 3
    assert results["node_success_1"].is_success is True
    assert results["node_success_1"].data_status == DataStatus.FRESH

    assert results["node_success_2"].is_success is True
    assert results["node_success_2"].data_status == DataStatus.FRESH

    # Failing node must be gracefully caught and marked UNAVAILABLE
    assert results["node_failing"].is_success is False
    assert results["node_failing"].data_status == DataStatus.UNAVAILABLE
    assert "Simulated remote connection abort" in results["node_failing"].error_message


def test_ingestion_service_status_reporting():
    """Verify get_location_status and get_all_statuses provide structured summaries."""
    service = MultiLocationIngestionService()
    # Query demo data for reference location
    service.fetch_location("mumbai_offshore", source="demo")

    status = service.get_location_status("mumbai_offshore", source="demo")
    assert status["location_id"] == "mumbai_offshore"
    assert status["is_reference"] is True
    assert status["validation_status"] == "reference_validated"
    assert status["data_status"] == "offline_demo"
    assert status["last_retrieval_time_utc"] is not None

    all_statuses = service.get_all_statuses(source="demo")
    assert len(all_statuses) == 8
    assert all_statuses[0]["location_id"] == "mumbai_offshore"


# =============================================================================
# 4. Backward Compatibility & Integration Tests
# =============================================================================

def test_predict_live_forecast_backward_compatibility(sample_raw_dataframe, trained_scaler):
    """Verify predict_live_forecast retains original signature and defaults to reference node."""
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )
    result = predict_live_forecast(
        prepared_df=df_window,
        anchor_time=actual_anchor,
        horizon=6,
        model_type="lstm",
        scaler=trained_scaler,
    )
    assert result.model_name == "WaveLSTM"
    assert result.horizon_hours == 6
    assert result.location_id == "mumbai_offshore"
    assert result.validation_status == "reference_validated"
    assert result.forecast_swh_m > 0.0
    assert result.sea_state_desc in [
        "Calm", "Smooth", "Slight", "Moderate", "Rough", "Very Rough", "High", "Very High", "Phenomenal"
    ]


def test_predict_live_forecast_non_reference_location(sample_raw_dataframe, trained_scaler):
    """Verify predict_live_forecast marks non-reference locations as experimental."""
    anchor = pd.Timestamp("2026-10-26 12:00:00", tz="UTC")
    df_window, actual_anchor = validate_and_prepare_input_window(
        sample_raw_dataframe, anchor_time=anchor, window_size=24
    )
    result = predict_live_forecast(
        prepared_df=df_window,
        anchor_time=actual_anchor,
        horizon=6,
        model_type="lstm",
        scaler=trained_scaler,
        location_id="goa_mormugao",
        validation_status="experimental_unvalidated",
    )
    assert result.location_id == "goa_mormugao"
    assert result.validation_status == "experimental_unvalidated"
    assert result.forecast_swh_m > 0.0


def test_fastapi_locations_endpoint():
    """Verify FastAPI GET /api/forecast/locations returns all 8 registered candidate nodes."""
    response = client.get("/api/forecast/locations")
    assert response.status_code == 200
    locations = response.json()

    assert isinstance(locations, list)
    assert len(locations) == 8

    # Reference node check
    ref_nodes = [loc for loc in locations if loc["is_reference"]]
    assert len(ref_nodes) == 1
    assert ref_nodes[0]["location_id"] == "mumbai_offshore"
    assert ref_nodes[0]["validation_status"] == "reference_validated"


def test_fastapi_live_endpoint_with_location():
    """Verify FastAPI GET /api/forecast/live accepts location_id and returns location metadata."""
    # Test with non-reference location
    response = client.get("/api/forecast/live?model=WaveLSTM&horizon=6&source=demo&location_id=chennai_port")
    assert response.status_code == 200
    data = response.json()

    assert data["location_id"] == "chennai_port"
    assert data["location_name"] == "Chennai Coastal Node"
    assert data["is_reference"] is False
    assert data["validation_status"] == "experimental_unvalidated"
    assert data["data_status"] == "offline_demo"
    assert "lstm" in data["predictions"]
    assert data["predictions"]["lstm"]["forecast_swh_m"] > 0.0

    # Test default backward compatibility (no location_id supplied)
    resp_default = client.get("/api/forecast/live?model=WaveLSTM&horizon=6&source=demo")
    assert resp_default.status_code == 200
    data_def = resp_default.json()
    assert data_def["location_id"] == "mumbai_offshore"
    assert data_def["is_reference"] is True
    assert data_def["validation_status"] == "reference_validated"
