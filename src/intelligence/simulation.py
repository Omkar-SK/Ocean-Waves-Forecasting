"""
Interactive Ocean Hazard Simulation Subsystem for OceanCast AI.

Generates realistic hypothetical marine weather hazards (severe swell events,
tropical cyclones, coastal storm surges, and moderate disturbances) across
the 8 registered Arabian Sea candidate coastal nodes.

GUARANTEES:
1. STRICT DATA ISOLATION: Simulated results are maintained in-memory and
   NEVER contaminate the persistent SQLite database (ocean_intelligence.db)
   or live Open-Meteo telemetry caches.
2. REPRODUCIBLE DYNAMICS: Realistic multi-hour storm evolutions with spatial
   decay from storm epicenters, dynamic alert generation, and heatmaps.
3. CLEAR LABELING: All generated outputs carry explicit simulation metadata
   and disclaimers distinguishing them from live operational observations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid

from backend.schemas.simulation import (
    HeatmapPoint,
    ScenarioMetadataItem,
    SimulatedTimeSeriesPoint,
    SimulationResultsResponse,
    SimulationStartRequest,
    SimulationStats,
    SimulationStatusResponse,
)
from src.intelligence.nodes import DEFAULT_COASTAL_NODES, CoastalNode
from src.intelligence.risk import RiskLevel


# -----------------------------------------------------------------------------
# Enums and Scenario Metadata
# -----------------------------------------------------------------------------

class ScenarioId(str, Enum):
    SEVERE_WAVE = "severe_wave"
    TROPICAL_CYCLONE = "tropical_cyclone"
    COASTAL_FLOODING = "coastal_flooding"
    MODERATE_DISTURBANCE = "moderate_disturbance"


@dataclass(frozen=True)
class ScenarioConfig:
    scenario_id: ScenarioId
    name: str
    category: str
    headline: str
    description: str
    severity_category: str
    coastal_impact_level: str
    recommended_action: str
    base_swh_m: float
    peak_swh_m: float
    peak_wind_kmh: float
    peak_mwp_s: float
    epicenter_lat: float
    epicenter_lon: float
    spatial_reach_km: float


SCENARIO_DEFINITIONS: Dict[ScenarioId, ScenarioConfig] = {
    ScenarioId.SEVERE_WAVE: ScenarioConfig(
        scenario_id=ScenarioId.SEVERE_WAVE,
        name="Severe Wave Conditions",
        category="High-Energy Swell Event",
        headline="Extreme Deep-Sea Swell & Destructive Wave Action (5.5–7.5m)",
        description=(
            "Simulates an intense oceanic storm generating long-period swell across the "
            "Arabian Sea with significant wave heights exceeding 6.5m and maximum wave crests "
            "up to 12.0m approaching the central and northern coastline."
        ),
        severity_category="CRITICAL",
        coastal_impact_level="Severe Offshore Wave Action",
        recommended_action=(
            "Complete prohibition of small craft operations, suspension of coastal port lighterage, "
            "and reinforcement of exposed harbor breakwaters."
        ),
        base_swh_m=5.50,
        peak_swh_m=7.40,
        peak_wind_kmh=85.0,
        peak_mwp_s=15.8,
        epicenter_lat=18.20,
        epicenter_lon=71.80,
        spatial_reach_km=480.0,
    ),
    ScenarioId.TROPICAL_CYCLONE: ScenarioConfig(
        scenario_id=ScenarioId.TROPICAL_CYCLONE,
        name="Tropical Cyclone Impact",
        category="Tropical Cyclonic Storm",
        headline="Approaching Category 3/4 Tropical Cyclone with Catastrophic Waves (7.0–10.0m)",
        description=(
            "Simulates an intense tropical cyclone tracking northeastward toward the Konkan and "
            "Gujarat coast, generating hurricane-force gusts (100–140 km/h), massive destructive seas "
            "up to 9.8m, and severe storm surge risk."
        ),
        severity_category="CRITICAL",
        coastal_impact_level="Catastrophic Storm Surge & Cyclone Landfall",
        recommended_action=(
            "Immediate coastal evacuation of low-lying promontories, full shutdown of commercial port "
            "terminals, and activation of emergency coastal defense sirens."
        ),
        base_swh_m=7.00,
        peak_swh_m=9.85,
        peak_wind_kmh=135.0,
        peak_mwp_s=17.5,
        epicenter_lat=17.20,
        epicenter_lon=71.50,
        spatial_reach_km=620.0,
    ),
    ScenarioId.COASTAL_FLOODING: ScenarioConfig(
        scenario_id=ScenarioId.COASTAL_FLOODING,
        name="Coastal Flooding & Surge",
        category="Storm Surge & Tidal Inundation",
        headline="Astronomical Spring Tide Inundation & Strong Onshore Winds (4.5–6.8m)",
        description=(
            "Simulates the confluence of high astronomical spring tides with persistent gale-force "
            "onshore winds, driving seawater penetration, estuary back-flow, and widespread inundation "
            "across vulnerable coastal settlements."
        ),
        severity_category="HIGH",
        coastal_impact_level="Severe Coastal Inundation Risk",
        recommended_action=(
            "Deployment of temporary coastal flood barriers, evacuation of beachfront properties, and "
            "continuous drainage pump operations across low-lying estuaries."
        ),
        base_swh_m=4.50,
        peak_swh_m=6.50,
        peak_wind_kmh=92.0,
        peak_mwp_s=13.8,
        epicenter_lat=18.60,
        epicenter_lon=72.30,
        spatial_reach_km=420.0,
    ),
    ScenarioId.MODERATE_DISTURBANCE: ScenarioConfig(
        scenario_id=ScenarioId.MODERATE_DISTURBANCE,
        name="Moderate Marine Disturbance",
        category="Monsoon Surge / Rough Sea State",
        headline="Elevated Monsoon Wind-Sea with Choppy Conditions (2.6–4.2m)",
        description=(
            "Simulates energetic southwesterly monsoon activity producing rough sea states, choppy "
            "wave-wind interaction, and cautionary navigation conditions for regional fisheries."
        ),
        severity_category="MODERATE",
        coastal_impact_level="Elevated Small-Craft Advisory",
        recommended_action=(
            "Advisory for artisanal fishing vessels to remain in harbor; recreational marine craft "
            "prohibited; commercial shipping advised to navigate with caution."
        ),
        base_swh_m=2.60,
        peak_swh_m=3.95,
        peak_wind_kmh=58.0,
        peak_mwp_s=10.8,
        epicenter_lat=15.80,
        epicenter_lon=72.80,
        spatial_reach_km=350.0,
    ),
}

DISCLAIMER_TEXT = (
    "Simulation results are hypothetical and are not actual marine observations or official "
    "meteorological warnings. Live database records remain untouched."
)

OFFSHORE_HEATMAP_GRID = [
    {"label": "Deep Arabian Sea Basin (North)", "region": "Offshore North", "lat": 19.50, "lon": 71.00},
    {"label": "West Mumbai Deep Shelf", "region": "Offshore Mumbai", "lat": 18.50, "lon": 71.40},
    {"label": "Central Arabian Sea Swell Core", "region": "Central Basin", "lat": 17.00, "lon": 71.00},
    {"label": "Konkan Offshore Trough", "region": "Offshore Konkan", "lat": 16.20, "lon": 72.00},
    {"label": "Goa Offshore Deep Ridge", "region": "Offshore Goa", "lat": 15.10, "lon": 72.40},
    {"label": "Malabar Deep Trench", "region": "Offshore Malabar", "lat": 13.50, "lon": 73.50},
    {"label": "Saurashtra Outer Shoal", "region": "Offshore Gujarat", "lat": 20.80, "lon": 70.20},
]


# -----------------------------------------------------------------------------
# Simulation Engine
# -----------------------------------------------------------------------------

class SimulationEngine:
    """
    In-memory simulation state machine managing storm evolution curves,
    spatial decays, dynamic alerts, and heatmaps without mutating persistent storage.
    """

    def __init__(self):
        self.is_active: bool = False
        self.state: str = "IDLE"  # "IDLE", "RUNNING", "PAUSED"
        self.current_scenario_id: Optional[ScenarioId] = None
        self.current_step: int = 6  # 0 to 24 (default initial hour: 6)
        self.total_steps: int = 24
        self.started_at_utc: Optional[datetime] = None
        self.updated_at_utc: Optional[datetime] = None
        self._cached_results: Optional[SimulationResultsResponse] = None

    def start(self, scenario_id: ScenarioId, initial_step: int = 6) -> SimulationResultsResponse:
        """Activate simulation mode with the selected scenario."""
        if scenario_id not in SCENARIO_DEFINITIONS:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")

        self.is_active = True
        self.state = "RUNNING"
        self.current_scenario_id = scenario_id
        self.current_step = max(0, min(24, int(initial_step)))
        now = datetime.now(timezone.utc)
        self.started_at_utc = now
        self.updated_at_utc = now

        self._cached_results = self._compute_simulation_results()
        return self._cached_results

    def pause(self) -> SimulationStatusResponse:
        """Pause timeline ticker."""
        if self.is_active:
            self.state = "PAUSED"
            self.updated_at_utc = datetime.now(timezone.utc)
        return self.get_status()

    def resume(self) -> SimulationStatusResponse:
        """Resume timeline ticker."""
        if self.is_active:
            self.state = "RUNNING"
            self.updated_at_utc = datetime.now(timezone.utc)
        return self.get_status()

    def step(self, delta: int = 1) -> SimulationResultsResponse:
        """Advance or step backward along the 24h timeline."""
        if not self.is_active or not self.current_scenario_id:
            # If idle, auto-start default Severe Wave scenario
            return self.start(ScenarioId.SEVERE_WAVE, initial_step=6)

        new_step = self.current_step + int(delta)
        if new_step > self.total_steps:
            new_step = self.total_steps
        elif new_step < 0:
            new_step = 0

        self.current_step = new_step
        self.updated_at_utc = datetime.now(timezone.utc)
        self._cached_results = self._compute_simulation_results()
        return self._cached_results

    def reset(self) -> SimulationStatusResponse:
        """Deactivate simulation mode and return to live operational data."""
        self.is_active = False
        self.state = "IDLE"
        self.current_scenario_id = None
        self.current_step = 6
        self.started_at_utc = None
        self.updated_at_utc = datetime.now(timezone.utc)
        self._cached_results = None
        return self.get_status()

    def get_status(self) -> SimulationStatusResponse:
        """Retrieve current simulation state."""
        cfg = SCENARIO_DEFINITIONS.get(self.current_scenario_id) if self.current_scenario_id else None
        elapsed = 0
        if self.started_at_utc:
            elapsed = int((datetime.now(timezone.utc) - self.started_at_utc).total_seconds() / 3600)

        return SimulationStatusResponse(
            is_active=self.is_active,
            state=self.state,
            scenario_id=self.current_scenario_id.value if self.current_scenario_id else None,
            scenario_name=cfg.name if cfg else None,
            current_step=self.current_step,
            total_steps=self.total_steps,
            elapsed_hours=elapsed,
            started_at_utc=self.started_at_utc.isoformat() if self.started_at_utc else None,
            updated_at_utc=self.updated_at_utc.isoformat() if self.updated_at_utc else None,
            disclaimer=DISCLAIMER_TEXT,
        )

    def get_results(self) -> SimulationResultsResponse:
        """Retrieve current results (computing if not cached)."""
        if not self.is_active or not self.current_scenario_id:
            return SimulationResultsResponse(
                status=self.get_status(),
                scenario=None,
                stats=None,
                nodes=[],
                heatmap_points=[],
                alerts=[],
                time_series=[],
            )

        if self._cached_results is None:
            self._cached_results = self._compute_simulation_results()
        return self._cached_results

    # -------------------------------------------------------------------------
    # Internal Calculation & Hazard Generation
    # -------------------------------------------------------------------------

    def _get_timeline_factor(self, step: int) -> float:
        """
        Calculates asymmetric storm intensity factor along the 24h timeline.
        Peaks at step 14 with smooth rise and decay.
        """
        peak_step = 14.0
        sigma = 5.5
        # Envelope ranges from ~0.66 at step 0 to 1.00 at step 14, and ~0.72 at step 24
        return 0.65 + 0.35 * math.exp(-((step - peak_step) ** 2) / (2.0 * (sigma ** 2)))

    def _distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Planar great-circle distance approximation in km."""
        dlat = (lat1 - lat2) * 111.0
        dlon = (lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
        return math.sqrt(dlat ** 2 + dlon ** 2)

    def _compute_simulation_results(self) -> SimulationResultsResponse:
        cfg = SCENARIO_DEFINITIONS[self.current_scenario_id]
        step = self.current_step
        timeline_factor = self._get_timeline_factor(step)
        now_utc = datetime.now(timezone.utc)
        step_time_utc = now_utc + timedelta(hours=step)

        # 1. Compute Node Conditions across the 8 registered coastal nodes
        nodes_list: List[Dict[str, Any]] = []
        alerts_list: List[Dict[str, Any]] = []
        max_swh = 0.0
        max_wind = 0.0
        critical_count = 0
        high_count = 0
        affected_regions_set = set()

        for node in DEFAULT_COASTAL_NODES:
            dist = self._distance_km(node.latitude, node.longitude, cfg.epicenter_lat, cfg.epicenter_lon)
            spatial_factor = max(0.38, math.exp(-dist / cfg.spatial_reach_km))

            # Calculate simulated significant wave height
            swh = cfg.base_swh_m + (cfg.peak_swh_m - cfg.base_swh_m) * timeline_factor * spatial_factor
            swh = round(float(swh), 2)
            if swh > max_swh:
                max_swh = swh

            # Wind speed
            wind_kmh = round(float(cfg.peak_wind_kmh * timeline_factor * spatial_factor), 1)
            if wind_kmh > max_wind:
                max_wind = wind_kmh

            # Max wave height (crest-to-trough extreme wave crest)
            max_wave_m = round(swh * 1.65, 2)

            # Wave period
            mwp = round(min(cfg.peak_mwp_s, 8.0 + 1.15 * swh), 1)

            # Determine Risk Level based on project thresholds:
            # Caution: >= 1.5m, High: >= 2.5m, Critical: >= 4.0m
            if swh >= 4.00:
                risk_level = RiskLevel.RED
                critical_count += 1
                affected_regions_set.add(node.region)
            elif swh >= 2.50:
                risk_level = RiskLevel.ORANGE
                high_count += 1
                affected_regions_set.add(node.region)
            elif swh >= 1.50:
                risk_level = RiskLevel.YELLOW
            else:
                risk_level = RiskLevel.GREEN

            # Multi-horizon synthetic forecasts (+1h, +6h, +24h)
            h1 = round(swh * self._get_timeline_factor(min(24, step + 1)) / max(0.01, timeline_factor), 2)
            h6 = round(swh * self._get_timeline_factor(min(24, step + 6)) / max(0.01, timeline_factor), 2)
            h24 = round(swh * self._get_timeline_factor(min(24, step + 24)) / max(0.01, timeline_factor), 2)

            # Generate dynamic alert if High or Critical
            node_alerts: List[Dict[str, Any]] = []
            if risk_level in (RiskLevel.RED, RiskLevel.ORANGE, RiskLevel.YELLOW):
                sev_label = "RED" if risk_level == RiskLevel.RED else "ORANGE" if risk_level == RiskLevel.ORANGE else "YELLOW"
                alert_obj = {
                    "alert_id": f"sim-{node.location_id}-{step}",
                    "fingerprint": f"sim-fp-{node.location_id}-{step}",
                    "location_id": node.location_id,
                    "location_name": node.name,
                    "event_type": "project_forecast_risk",
                    "severity": sev_label,
                    "status": "ACTIVE",
                    "trigger_reason": (
                        f"Simulated Significant Wave Height ({swh:.2f}m) exceeds "
                        f"{'Critical (4.00m)' if swh >= 4.0 else 'High (2.50m)' if swh >= 2.5 else 'Caution (1.50m)'} threshold."
                    ),
                    "predicted_swh_m": swh,
                    "threshold_m": 4.0 if swh >= 4.0 else 2.5 if swh >= 2.5 else 1.5,
                    "forecast_horizon_hours": 6,
                    "expected_event_time_utc": step_time_utc.isoformat(),
                    "anchor_time_utc": now_utc.isoformat(),
                    "forecast_issue_time_utc": now_utc.isoformat(),
                    "assessment_time_utc": now_utc.isoformat(),
                    "source_name": "OceanCast Hazard Simulation Engine",
                    "data_status": "SIMULATED_DATA",
                    "model_name": "WaveLSTM (Simulated)",
                    "validation_status": "simulated",
                    "is_reference": node.is_reference,
                    "created_at_utc": now_utc.isoformat(),
                    "updated_at_utc": now_utc.isoformat(),
                    "is_simulated": True,
                    "risk_score": min(99, int(30 + swh * 7.0 + wind_kmh * 0.15)),
                    "explanation": (
                        f"Simulated {cfg.name} active. High sea states ({swh:.2f}m) with peak wave periods "
                        f"of {mwp:.1f}s and wind gusts of {wind_kmh:.1f} km/h generate severe coastal hazard exposure."
                    ),
                    "recommended_action": cfg.recommended_action,
                    "operator_notes": f"Scenario: {cfg.name} (T+{step}h timeline).",
                }
                node_alerts.append(alert_obj)
                alerts_list.append(alert_obj)

            node_dict = {
                "location_id": node.location_id,
                "name": node.name,
                "latitude": node.latitude,
                "longitude": node.longitude,
                "coordinate_label": node.coordinate_label,
                "region": node.region,
                "is_reference": node.is_reference,
                "validation_status": "simulated",
                "data_status": "SIMULATED_DATA",
                "data_source": "Hazard Simulation Mode",
                "retrieval_time_utc": now_utc.isoformat(),
                "forecast_issue_time_utc": now_utc.isoformat(),
                "latest_obs_swh_m": swh,
                "latest_obs_time_utc": now_utc.isoformat(),
                "forecast_swh_m": h6,
                "forecast_horizon_hours": 6,
                "forecast_valid_time_utc": (now_utc + timedelta(hours=6)).isoformat(),
                "max_window_swh_m": max_wave_m,
                "current_risk_level": risk_level.value,
                "highest_risk_level": risk_level.value,
                "marker_hex_color": risk_level.hex_color,
                "threshold_crossed_m": 4.0 if swh >= 4.0 else 2.5 if swh >= 2.5 else 1.5,
                "active_alert_count": len(node_alerts),
                "active_alerts": node_alerts,
                "is_degraded": False,
                "trigger_explanation": f"Simulated {cfg.name} (T+{step}h): Hs = {swh:.2f}m, Wind = {wind_kmh:.1f}km/h.",
                "limitations": [
                    "Hypothetical simulation dataset for training & verification demonstration.",
                    "Not an actual live marine observation or official disaster warning.",
                ],
                "multi_horizon_predictions": {1: h1, 6: h6, 24: h24},
                "external_nwp_forecasts": {1: round(h1 * 0.96, 2), 6: round(h6 * 0.95, 2), 24: round(h24 * 0.94, 2)},
            }
            nodes_list.append(node_dict)

        # 2. Build Spatially Varying Heatmap Points (Coastal Nodes + Offshore Core Plumes)
        heatmap_points: List[HeatmapPoint] = []
        all_grid_points = [
            {"label": n["name"], "region": n["region"], "lat": n["latitude"], "lon": n["longitude"]}
            for n in nodes_list
        ] + OFFSHORE_HEATMAP_GRID

        for pt in all_grid_points:
            dist = self._distance_km(pt["lat"], pt["lon"], cfg.epicenter_lat, cfg.epicenter_lon)
            spatial_factor = max(0.35, math.exp(-dist / cfg.spatial_reach_km))
            pt_swh = round(float(cfg.base_swh_m + (cfg.peak_swh_m - cfg.base_swh_m) * timeline_factor * spatial_factor), 2)
            pt_wind = round(float(cfg.peak_wind_kmh * timeline_factor * spatial_factor), 1)
            pt_mwp = round(min(cfg.peak_mwp_s, 8.0 + 1.15 * pt_swh), 1)
            pt_max_wave = round(pt_swh * 1.65, 2)

            intensity = max(0.15, min(1.0, pt_swh / 10.0))

            if pt_swh >= 4.0:
                rl = "CRITICAL"
                color = "#ef4444"
                radius = 55.0
            elif pt_swh >= 2.5:
                rl = "HIGH"
                color = "#f97316"
                radius = 42.0
            elif pt_swh >= 1.5:
                rl = "MODERATE"
                color = "#f59e0b"
                radius = 32.0
            else:
                rl = "LOW"
                color = "#10b981"
                radius = 24.0

            heatmap_points.append(
                HeatmapPoint(
                    latitude=pt["lat"],
                    longitude=pt["lon"],
                    label=pt["label"],
                    region=pt["region"],
                    swh_m=pt_swh,
                    max_wave_m=pt_max_wave,
                    mwp_s=pt_mwp,
                    wind_kmh=pt_wind,
                    wind_dir_deg=round((dist * 0.85 + step * 12.0) % 360, 1),
                    intensity=round(intensity, 3),
                    risk_level=rl,
                    color=color,
                    radius_km=radius,
                )
            )

        # 3. Overall Composite Coastal Risk Index (0-100)
        risk_index = min(99, max(25, int((max_swh / 10.0) * 65.0 + (max_wind / 140.0) * 35.0)))

        stats = SimulationStats(
            active_alerts=len(alerts_list),
            critical_risk_zones=critical_count,
            high_risk_zones=high_count,
            max_swh_m=round(max_swh, 2),
            max_wind_kmh=round(max_wind, 1),
            coastal_risk_index=risk_index,
            affected_regions=sorted(list(affected_regions_set)),
            scenario_name=cfg.name,
            severity_category=cfg.severity_category,
            headline=cfg.headline,
            coastal_impact_level=cfg.coastal_impact_level,
            recommended_action=cfg.recommended_action,
        )

        # 4. Generate 24-Hour Time-Series Progression
        time_series: List[SimulatedTimeSeriesPoint] = []
        for s in range(self.total_steps + 1):
            s_time_factor = self._get_timeline_factor(s)
            ref_swh = round(float(cfg.base_swh_m + (cfg.peak_swh_m - cfg.base_swh_m) * s_time_factor * 0.95), 2)
            ref_wind = round(float(cfg.peak_wind_kmh * s_time_factor * 0.95), 1)
            ref_mwp = round(min(cfg.peak_mwp_s, 8.0 + 1.15 * ref_swh), 1)

            # Model variations
            lstm_pred = round(ref_swh * (1.0 + 0.02 * math.sin(s / 2.0)), 2)
            ann_pred = round(ref_swh * (1.0 + 0.04 * math.cos(s / 3.0)), 2)
            cnn_pred = round(ref_swh * (1.0 - 0.03 * math.sin(s / 2.5)), 2)
            pers_val = round(ref_swh * 0.92, 2)

            is_forecast = bool(s > step)
            actual_val = ref_swh if not is_forecast else None

            rl_s = "CRITICAL" if ref_swh >= 4.0 else "HIGH" if ref_swh >= 2.5 else "MODERATE" if ref_swh >= 1.5 else "LOW"

            t_str = f"T+{s:02d}h"
            t_iso = (now_utc + timedelta(hours=s)).isoformat()

            time_series.append(
                SimulatedTimeSeriesPoint(
                    step=s,
                    time=t_str,
                    timestamp=t_iso,
                    actual=actual_val,
                    lstm=lstm_pred,
                    ann=ann_pred,
                    cnn1d=cnn_pred,
                    persistence=pers_val,
                    wind_kmh=ref_wind,
                    mwp_s=ref_mwp,
                    risk_level=rl_s,
                    isForecast=is_forecast,
                )
            )

        scenario_meta = ScenarioMetadataItem(
            scenario_id=cfg.scenario_id.value,
            name=cfg.name,
            category=cfg.category,
            headline=cfg.headline,
            description=cfg.description,
            severity_category=cfg.severity_category,
            coastal_impact_level=cfg.coastal_impact_level,
            recommended_action=cfg.recommended_action,
            base_swh_m=cfg.base_swh_m,
            peak_swh_m=cfg.peak_swh_m,
            peak_wind_kmh=cfg.peak_wind_kmh,
            peak_mwp_s=cfg.peak_mwp_s,
        )

        return SimulationResultsResponse(
            status=self.get_status(),
            scenario=scenario_meta,
            stats=stats,
            nodes=nodes_list,
            heatmap_points=heatmap_points,
            alerts=alerts_list,
            time_series=time_series,
        )


# Global Singleton Instance
_simulation_engine_instance: Optional[SimulationEngine] = None


def get_simulation_engine() -> SimulationEngine:
    """Retrieve or initialize the global simulation engine singleton."""
    global _simulation_engine_instance
    if _simulation_engine_instance is None:
        _simulation_engine_instance = SimulationEngine()
    return _simulation_engine_instance
