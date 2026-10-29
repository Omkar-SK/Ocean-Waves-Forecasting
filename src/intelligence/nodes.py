"""
Coastal Node Registry for Real-Time Ocean Intelligence.

Manages geographic coordinates, coastal regional metadata, and model applicability
states for coastal monitoring points across the Indian coastline.

IMPORTANT GEOGRAPHIC NOTICE:
Coordinates in this registry represent candidate offshore numerical forecasting grid cells
(~5 to 25 km offshore from major coastal ports and promontories). They must NOT be represented
as physically moored oceanographic buoy stations unless verified against specific institutional
metadata (such as MoES/INCOIS Ocean Observation System moored buoy deployments).

Only the Mumbai Offshore node (18.50°N, 72.50°E) is the calibrated reference location for
the existing trained PyTorch checkpoints (WaveANN, WaveLSTM, WaveCNN1D) and scaler.pkl.
All other locations are explicitly designated as 'EXPERIMENTAL_UNVALIDATED' regional transfers.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional


class ValidationStatus(str, Enum):
    """Geographic and climatological model calibration status."""
    REFERENCE_VALIDATED = "reference_validated"
    EXPERIMENTAL_UNVALIDATED = "experimental_unvalidated"


@dataclass(frozen=True)
class CoastalNode:
    """
    Immutable representation of a coastal forecasting monitoring node.
    """
    location_id: str
    name: str
    latitude: float
    longitude: float
    region: str
    timezone_str: str = "Asia/Kolkata"
    enabled: bool = True
    is_reference: bool = False
    validation_status: ValidationStatus = ValidationStatus.EXPERIMENTAL_UNVALIDATED
    description: str = ""
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Validate location_id format
        if not self.location_id or not isinstance(self.location_id, str):
            raise ValueError("location_id must be a non-empty string.")
        if not re.match(r"^[a-z0-9_]+$", self.location_id):
            raise ValueError(
                f"Invalid location_id '{self.location_id}'. Must be lowercase alphanumeric with underscores."
            )

        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string.")

        # Validate coordinates
        if not isinstance(self.latitude, (int, float)) or not (-90.0 <= float(self.latitude) <= 90.0):
            raise ValueError(f"Latitude {self.latitude} is out of valid bounds [-90.0, 90.0].")
        if not isinstance(self.longitude, (int, float)) or not (-180.0 <= float(self.longitude) <= 180.0):
            raise ValueError(f"Longitude {self.longitude} is out of valid bounds [-180.0, 180.0].")

        # Consistency check for reference status
        if self.is_reference and self.validation_status != ValidationStatus.REFERENCE_VALIDATED:
            raise ValueError("Reference location must have validation_status = REFERENCE_VALIDATED.")

    @property
    def coordinate_label(self) -> str:
        """Formatted string of geographic coordinates."""
        lat_dir = "N" if self.latitude >= 0 else "S"
        lon_dir = "E" if self.longitude >= 0 else "W"
        return f"{abs(self.latitude):.2f}°{lat_dir}, {abs(self.longitude):.2f}°{lon_dir}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert node attributes to serializable dictionary."""
        return {
            "location_id": self.location_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "coordinate_label": self.coordinate_label,
            "region": self.region,
            "timezone": self.timezone_str,
            "enabled": self.enabled,
            "is_reference": self.is_reference,
            "validation_status": self.validation_status.value,
            "description": self.description,
            "notes": list(self.notes),
        }


class CoastalNodeRegistry:
    """
    Configurable registry for managing coastal monitoring locations.
    """

    def __init__(self, initial_nodes: Optional[List[CoastalNode]] = None):
        self._nodes: Dict[str, CoastalNode] = {}
        if initial_nodes:
            for node in initial_nodes:
                self.register(node)

    def register(self, node: CoastalNode) -> None:
        """Register a new coastal node. Raises ValueError if location_id already exists."""
        if not isinstance(node, CoastalNode):
            raise TypeError("node must be an instance of CoastalNode.")
        if node.location_id in self._nodes:
            raise ValueError(f"Coastal node with location_id '{node.location_id}' already registered.")
        self._nodes[node.location_id] = node

    def get(self, location_id: str) -> CoastalNode:
        """Retrieve a registered coastal node by its location_id."""
        if location_id not in self._nodes:
            raise KeyError(
                f"Location ID '{location_id}' not found in registry. "
                f"Available nodes: {list(self._nodes.keys())}"
            )
        return self._nodes[location_id]

    def get_reference_node(self) -> CoastalNode:
        """Retrieve the primary calibrated reference location (Mumbai Offshore)."""
        for node in self._nodes.values():
            if node.is_reference:
                return node
        raise RuntimeError("No reference location found in coastal node registry.")

    def list_all(self, enabled_only: bool = False) -> List[CoastalNode]:
        """List all registered coastal nodes."""
        if enabled_only:
            return [node for node in self._nodes.values() if node.enabled]
        return list(self._nodes.values())

    def set_enabled(self, location_id: str, enabled: bool) -> CoastalNode:
        """Enable or disable a location in the registry."""
        node = self.get(location_id)
        if node.is_reference and not enabled:
            raise ValueError("The primary reference location cannot be disabled.")

        updated = CoastalNode(
            location_id=node.location_id,
            name=node.name,
            latitude=node.latitude,
            longitude=node.longitude,
            region=node.region,
            timezone_str=node.timezone_str,
            enabled=enabled,
            is_reference=node.is_reference,
            validation_status=node.validation_status,
            description=node.description,
            notes=node.notes,
        )
        self._nodes[location_id] = updated
        return updated

    def to_dict_list(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Return list of serialized dictionaries for UI and API consumers."""
        return [node.to_dict() for node in self.list_all(enabled_only=enabled_only)]

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, location_id: str) -> bool:
        return location_id in self._nodes


# -----------------------------------------------------------------------------
# Default Candidate Coastal Monitoring Nodes (Indian Coastline)
# -----------------------------------------------------------------------------
DEFAULT_COASTAL_NODES: List[CoastalNode] = [
    CoastalNode(
        location_id="mumbai_offshore",
        name="Mumbai Offshore (Reference Node)",
        latitude=18.50,
        longitude=72.50,
        region="Maharashtra Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=True,
        validation_status=ValidationStatus.REFERENCE_VALIDATED,
        description="Verified deep Arabian Sea offshore node (~80km WSW of Mumbai).",
        notes=[
            "Exact ERA5 reanalysis training grid node (2012-2026, 129,696 hours).",
            "Fitted reference for models/scaler.pkl and all 9 trained PyTorch checkpoints.",
            "Benchmarked against Naive Persistence and NWP across 1h, 6h, and 24h horizons.",
        ],
    ),
    CoastalNode(
        location_id="goa_mormugao",
        name="Mormugao / Goa Coastal Node",
        latitude=15.40,
        longitude=73.70,
        region="Goa / Central Arabian Sea",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Offshore waters (~15km west of Mormugao Port, Goa).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "Subject to seasonal summer monsoon swell and coastal bathymetric shoaling.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
    CoastalNode(
        location_id="mangalore_port",
        name="Mangalore Coastal Node",
        latitude=12.90,
        longitude=74.75,
        region="Karnataka Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Offshore waters (~12km west of New Mangalore Port).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
    CoastalNode(
        location_id="kochi_coast",
        name="Kochi Coastal Node",
        latitude=9.95,
        longitude=76.15,
        region="Kerala Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Offshore waters (~10km west of Kochi Port entrance).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "Influenced by intense southwest monsoon wind-waves and Indian Ocean distant swell.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
    CoastalNode(
        location_id="kanyakumari",
        name="Kanyakumari Promontory Node",
        latitude=8.05,
        longitude=77.55,
        region="Tamil Nadu / Cape Comorin",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Southern tip tri-junction (Arabian Sea, Bay of Bengal, Indian Ocean).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "Complex cross-sea conditions and year-round high-energy Southern Ocean swell exposure.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
    CoastalNode(
        location_id="chennai_port",
        name="Chennai Coastal Node",
        latitude=13.10,
        longitude=80.35,
        region="Tamil Nadu / Coromandel Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Offshore waters (~8km east of Chennai Port in the Bay of Bengal).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "Subject to Northeast Monsoon (October-December) cyclone and storm surge events.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
    CoastalNode(
        location_id="visakhapatnam_port",
        name="Visakhapatnam Coastal Node",
        latitude=17.65,
        longitude=83.35,
        region="Andhra Pradesh Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Offshore waters (~6km east of Dolphin's Nose, Visakhapatnam).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "Deep natural coastal bathymetry in the Western Bay of Bengal.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
    CoastalNode(
        location_id="paradip_port",
        name="Paradip Coastal Node",
        latitude=20.25,
        longitude=86.70,
        region="Odisha Coast",
        timezone_str="Asia/Kolkata",
        enabled=True,
        is_reference=False,
        validation_status=ValidationStatus.EXPERIMENTAL_UNVALIDATED,
        description="Offshore waters (~8km south-east of Paradip Port).",
        notes=[
            "Candidate forecast coordinates, not a verified buoy or monitoring station.",
            "Frequent cyclonic depression track landfall zone in North Bay of Bengal.",
            "DL predictions represent zero-shot regional transfer without localized calibration.",
        ],
    ),
]


def create_default_registry() -> CoastalNodeRegistry:
    """Instantiate a registry populated with the 8 validated candidate coastal nodes."""
    return CoastalNodeRegistry(DEFAULT_COASTAL_NODES)


# Singleton instance for consistent use across backend and frontend
COASTAL_NODE_REGISTRY = create_default_registry()


def get_coastal_registry() -> CoastalNodeRegistry:
    """Accessor for the global coastal node registry."""
    return COASTAL_NODE_REGISTRY
