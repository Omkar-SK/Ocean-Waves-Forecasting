"""
API router for the Interactive Ocean Hazard Simulation Subsystem.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.simulation import (
    ScenarioMetadataItem,
    SimulationResultsResponse,
    SimulationStartRequest,
    SimulationStatusResponse,
)
from src.intelligence.simulation import (
    SCENARIO_DEFINITIONS,
    ScenarioId,
    get_simulation_engine,
)

router = APIRouter(prefix="/api/simulation", tags=["Hazard Simulation Mode"])


@router.get("/scenarios", response_model=List[ScenarioMetadataItem])
def list_available_scenarios():
    """Retrieve catalog of predefined hypothetical ocean hazard scenarios."""
    items = []
    for cfg in SCENARIO_DEFINITIONS.values():
        items.append(
            ScenarioMetadataItem(
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
        )
    return items


@router.post("/start", response_model=SimulationResultsResponse)
def start_simulation(payload: SimulationStartRequest):
    """
    Activate Simulation Mode with the selected scenario.
    Generates synthetic high-risk wave conditions, alerts, and spatial heatmaps.
    """
    try:
        scenario_id_enum = ScenarioId(payload.scenario_id)
    except ValueError:
        valid_ids = [s.value for s in ScenarioId]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario_id '{payload.scenario_id}'. Must be one of: {valid_ids}",
        )

    engine = get_simulation_engine()
    results = engine.start(scenario_id_enum, initial_step=payload.initial_step)
    return results


@router.post("/pause", response_model=SimulationStatusResponse)
def pause_simulation():
    """Pause the simulation timeline ticker."""
    engine = get_simulation_engine()
    return engine.pause()


@router.post("/resume", response_model=SimulationStatusResponse)
def resume_simulation():
    """Resume the paused simulation timeline ticker."""
    engine = get_simulation_engine()
    return engine.resume()


@router.post("/step", response_model=SimulationResultsResponse)
def step_simulation_timeline(delta: int = Query(1, description="Hour step delta (+1 forward, -1 backward)")):
    """Step the simulation along the 24-hour storm progression timeline."""
    engine = get_simulation_engine()
    return engine.step(delta)


@router.post("/reset", response_model=SimulationStatusResponse)
def reset_simulation():
    """
    Deactivate simulation mode and immediately restore live operational data.
    Removes all synthetic heatmaps, simulated alerts, and hazard states.
    """
    engine = get_simulation_engine()
    return engine.reset()


@router.get("/status", response_model=SimulationStatusResponse)
def get_simulation_status():
    """Retrieve current simulation mode status and active scenario."""
    engine = get_simulation_engine()
    return engine.get_status()


@router.get("/results", response_model=SimulationResultsResponse)
def get_simulation_results():
    """Retrieve active simulation results including nodes, heatmaps, alerts, and forecast."""
    engine = get_simulation_engine()
    return engine.get_results()

