"""
Multi-Location Ingestion Service for Real-Time Ocean Intelligence.

Orchestrates concurrent, fault-isolated data retrieval across registered coastal nodes
with in-memory TTL caching, quality gating, and explicit data freshness tracking.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.intelligence.nodes import (
    CoastalNode,
    CoastalNodeRegistry,
    ValidationStatus,
    get_coastal_registry,
)
from src.intelligence.provider import (
    BaseMarineDataProvider,
    DataStatus,
    OfflineDemoProvider,
    OpenMeteoProvider,
    ProviderFetchResult,
)

logger = logging.getLogger("OceanIntelligence.Ingestion")


class MultiLocationIngestionService:
    """
    Service managing multi-location coastal telemetry ingestion.

    Guarantees:
    - Reusable marine provider abstraction (Open-Meteo REST & Offline ERA5 Demo).
    - In-memory TTL caching (default 15 minutes) preventing redundant API queries.
    - Preserves source retrieval timestamps and metadata when serving cached responses.
    - Per-location error isolation: failure of one location does not disrupt others.
    - Bounded concurrency with configurable thread pool (default max 4 workers).
    - Distinction between retrieval time, forecast cycle issue time, and valid data times.
    """

    def __init__(
        self,
        registry: Optional[CoastalNodeRegistry] = None,
        live_provider: Optional[BaseMarineDataProvider] = None,
        demo_provider: Optional[BaseMarineDataProvider] = None,
        cache_ttl_s: int = 900,
        max_workers: int = 4,
    ):
        self.registry = registry or get_coastal_registry()
        self.live_provider = live_provider or OpenMeteoProvider()
        self.demo_provider = demo_provider or OfflineDemoProvider()
        self.cache_ttl_s = cache_ttl_s
        self.max_workers = max_workers
        self._cache: Dict[str, Tuple[ProviderFetchResult, float]] = {}

    def _cache_key(self, location_id: str, source: str) -> str:
        return f"{location_id}_{source.lower()}"

    def fetch_location(
        self,
        location_id: str,
        source: str = "live",
        force_refresh: bool = False,
        timeout_s: int = 10,
    ) -> ProviderFetchResult:
        """
        Fetch telemetry for a single registered coastal node with caching.

        Parameters
        ----------
        location_id : str
            Identifier of registered coastal node.
        source : str
            'live' for Open-Meteo REST API or 'demo' for offline ERA5 archive.
        force_refresh : bool
            If True, bypasses cache and forces network query.
        timeout_s : int
            Network timeout limit.

        Returns
        -------
        ProviderFetchResult
            Standardized result envelope.
        """
        node = self.registry.get(location_id)
        c_key = self._cache_key(location_id, source)
        now_ts = time.time()

        # Check in-memory cache
        if not force_refresh and c_key in self._cache:
            cached_res, cached_at = self._cache[c_key]
            age_s = now_ts - cached_at
            if age_s < self.cache_ttl_s and cached_res.is_success:
                # Return cached result with CACHED status, preserving original retrieval_time_utc!
                return ProviderFetchResult(
                    provider_name=cached_res.provider_name,
                    location_id=cached_res.location_id,
                    latitude=cached_res.latitude,
                    longitude=cached_res.longitude,
                    retrieval_time_utc=cached_res.retrieval_time_utc,
                    data_status=DataStatus.CACHED if not cached_res.is_demo else DataStatus.OFFLINE_DEMO,
                    raw_df=cached_res.raw_df.copy(),
                    forecast_issue_time_utc=cached_res.forecast_issue_time_utc,
                    valid_time_start_utc=cached_res.valid_time_start_utc,
                    valid_time_end_utc=cached_res.valid_time_end_utc,
                    latest_obs_time_utc=cached_res.latest_obs_time_utc,
                    is_forecast=cached_res.is_forecast,
                    is_demo=cached_res.is_demo,
                    external_nwp_forecasts=dict(cached_res.external_nwp_forecasts),
                    error_message=cached_res.error_message,
                    attribution=cached_res.attribution,
                    notes=list(cached_res.notes) + [f"Served from in-memory cache (age: {int(age_s)}s)."],
                )

        # Select provider based on source
        provider = self.demo_provider if source.lower() == "demo" else self.live_provider

        try:
            res = provider.fetch_telemetry(node=node, timeout_s=timeout_s)
        except Exception as e:
            logger.error("Unhandled error querying provider for node %s: %s", location_id, e)
            res = ProviderFetchResult(
                provider_name=provider.provider_name,
                location_id=node.location_id,
                latitude=node.latitude,
                longitude=node.longitude,
                retrieval_time_utc=datetime.now(timezone.utc),
                data_status=DataStatus.UNAVAILABLE,
                raw_df=pd.DataFrame(),
                forecast_issue_time_utc=None,
                valid_time_start_utc=None,
                valid_time_end_utc=None,
                latest_obs_time_utc=None,
                is_forecast=False,
                is_demo=(source.lower() == "demo"),
                external_nwp_forecasts={},
                error_message=str(e),
                attribution=provider.attribution,
                notes=[f"Exception in fetch_telemetry: {e}"],
            )

        # Cache only successful responses
        if res.is_success:
            self._cache[c_key] = (res, now_ts)

        return res

    def fetch_all_locations(
        self,
        source: str = "live",
        enabled_only: bool = True,
        force_refresh: bool = False,
        timeout_s: int = 10,
    ) -> Dict[str, ProviderFetchResult]:
        """
        Fetch telemetry for all registered coastal nodes with bounded concurrency and error isolation.

        Parameters
        ----------
        source : str
            'live' or 'demo'.
        enabled_only : bool
            Whether to only query active locations.
        force_refresh : bool
            Force network queries bypassing cache.
        timeout_s : int
            Network timeout limit.

        Returns
        -------
        Dict[str, ProviderFetchResult]
            Dictionary mapping location_id to its ProviderFetchResult.
        """
        nodes = self.registry.list_all(enabled_only=enabled_only)
        results: Dict[str, ProviderFetchResult] = {}

        if not nodes:
            return results

        # Single-threaded path for small counts or demo
        if len(nodes) == 1:
            n = nodes[0]
            results[n.location_id] = self.fetch_location(
                location_id=n.location_id,
                source=source,
                force_refresh=force_refresh,
                timeout_s=timeout_s,
            )
            return results

        # Bounded concurrency across locations
        workers = min(len(nodes), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_loc = {
                executor.submit(
                    self.fetch_location,
                    node.location_id,
                    source,
                    force_refresh,
                    timeout_s,
                ): node.location_id
                for node in nodes
            }

            for future in as_completed(future_to_loc):
                loc_id = future_to_loc[future]
                try:
                    res = future.result()
                    results[loc_id] = res
                except Exception as exc:
                    logger.error("Exception during concurrent fetch for %s: %s", loc_id, exc)
                    node = self.registry.get(loc_id)
                    results[loc_id] = ProviderFetchResult(
                        provider_name="Unknown",
                        location_id=loc_id,
                        latitude=node.latitude,
                        longitude=node.longitude,
                        retrieval_time_utc=datetime.now(timezone.utc),
                        data_status=DataStatus.UNAVAILABLE,
                        raw_df=pd.DataFrame(),
                        error_message=f"Concurrency execution error: {exc}",
                        is_demo=(source.lower() == "demo"),
                    )

        return results

    def get_location_status(self, location_id: str, source: str = "live") -> Dict[str, Any]:
        """Return structured summary status for a specific node."""
        node = self.registry.get(location_id)
        c_key = self._cache_key(location_id, source)
        is_cached = c_key in self._cache

        latest_fetch = None
        data_status = DataStatus.UNAVAILABLE.value
        freshness_mins = None
        latest_obs_str = None
        issue_time_str = None
        error_msg = None

        if is_cached:
            res, _ = self._cache[c_key]
            latest_fetch = res.retrieval_time_utc.isoformat()
            data_status = res.data_status.value
            freshness_mins = res.freshness_minutes
            if res.latest_obs_time_utc:
                latest_obs_str = res.latest_obs_time_utc.isoformat()
            if res.forecast_issue_time_utc:
                issue_time_str = res.forecast_issue_time_utc.isoformat()
            error_msg = res.error_message

        return {
            "location_id": node.location_id,
            "name": node.name,
            "coordinate_label": node.coordinate_label,
            "latitude": node.latitude,
            "longitude": node.longitude,
            "region": node.region,
            "is_reference": node.is_reference,
            "validation_status": node.validation_status.value,
            "enabled": node.enabled,
            "data_status": data_status,
            "last_retrieval_time_utc": latest_fetch,
            "latest_observation_time_utc": latest_obs_str,
            "forecast_issue_time_utc": issue_time_str,
            "freshness_minutes": freshness_mins,
            "last_error": error_msg,
        }

    def get_all_statuses(self, source: str = "live", enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Return structured status list for all nodes in registry."""
        nodes = self.registry.list_all(enabled_only=enabled_only)
        return [self.get_location_status(n.location_id, source=source) for n in nodes]

    def clear_cache(self, location_id: Optional[str] = None):
        """Clear cache for all or a specific location."""
        if location_id:
            for k in list(self._cache.keys()):
                if k.startswith(f"{location_id}_"):
                    del self._cache[k]
        else:
            self._cache.clear()


# Singleton instance
INGESTION_SERVICE = MultiLocationIngestionService()


def get_ingestion_service() -> MultiLocationIngestionService:
    """Accessor for the global multi-location ingestion service."""
    return INGESTION_SERVICE
