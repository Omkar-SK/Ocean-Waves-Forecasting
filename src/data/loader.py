"""
NetCDF Dataset Loading and Dynamic Variable Discovery for ERA5 Reanalysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import xarray as xr


# Standard canonical variable concepts we look for in ERA5
TARGET_VARIABLE_CONCEPTS = {
    "swh": {
        "description": "Significant wave height of combined wind waves and swell",
        "expected_units": ["m", "meters"],
        "aliases": ["swh", "significant_wave_height", "hs"],
    },
    "mwp": {
        "description": "Mean wave period",
        "expected_units": ["s", "seconds"],
        "aliases": ["mwp", "mean_wave_period", "tp", "tm"],
    },
    "mwd": {
        "description": "Mean wave direction",
        "expected_units": ["Degree true", "degrees", "deg", "degree"],
        "aliases": ["mwd", "mean_wave_direction"],
    },
    "u10": {
        "description": "10m zonal U wind component",
        "expected_units": ["m s**-1", "m/s", "m s-1"],
        "aliases": ["u10", "10u", "u_wind_10m"],
    },
    "v10": {
        "description": "10m meridional V wind component",
        "expected_units": ["m s**-1", "m/s", "m s-1"],
        "aliases": ["v10", "10v", "v_wind_10m"],
    },
}


def load_netcdf_dataset(file_path: Union[str, Path]) -> xr.Dataset:
    """
    Open an ERA5 NetCDF file with automated backend engine fallback (h5netcdf -> netcdf4 -> scipy).

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the NetCDF file (.nc).

    Returns
    -------
    xr.Dataset
        Loaded xarray dataset.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    last_err = None
    for engine in ["h5netcdf", "netcdf4", "scipy"]:
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            last_err = e
            continue

    try:
        return xr.open_dataset(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open NetCDF file {path} with any supported engine: {e}") from last_err


def inspect_netcdf_variables(ds: xr.Dataset) -> Dict[str, Dict[str, Any]]:
    """
    Inspect all data variables in a dataset, extracting their names, units, shapes, and attributes.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of variable inspection metadata.
    """
    var_info = {}
    for var_name, da in ds.data_vars.items():
        var_info[var_name] = {
            "name": str(var_name),
            "long_name": str(da.attrs.get("long_name", da.attrs.get("standard_name", "N/A"))),
            "units": str(da.attrs.get("units", "dimensionless")),
            "shape": list(da.shape),
            "dims": list(da.dims),
            "dtype": str(da.dtype),
            "attrs": dict(da.attrs),
        }
    return var_info


def discover_era5_datasets(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Discover all NetCDF files in data_dir, inspect their actual variables,
    detect overlaps, and map canonical ERA5 variables to their containing files.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing NetCDF files.

    Returns
    -------
    Dict[str, Any]
        Mapping containing:
        - 'files': Dict[str, Dict] with file details and datasets
        - 'variable_mapping': Dict[var_name -> file_path]
        - 'overlapping_variables': Dict[var_name -> list of files]
        - 'identified_concepts': Dict[canonical_name -> actual_var_name]
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {data_path}")

    nc_files = sorted(list(data_path.glob("*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in directory: {data_path}")

    files_info = {}
    var_to_files = {}

    for f in nc_files:
        ds = load_netcdf_dataset(f)
        var_meta = inspect_netcdf_variables(ds)
        time_coord = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)

        files_info[f.name] = {
            "path": f,
            "dataset": ds,
            "variables": var_meta,
            "time_coord": time_coord,
            "coords": list(ds.coords.keys()),
            "dims": dict(ds.sizes),
        }

        for var_name in var_meta.keys():
            var_to_files.setdefault(var_name, []).append(f.name)

    # Detect overlaps
    overlapping = {v: flist for v, flist in var_to_files.items() if len(flist) > 1}

    # Map canonical concepts to actual variables and source files
    concept_mapping = {}
    for concept_id, concept_info in TARGET_VARIABLE_CONCEPTS.items():
        found = False
        aliases = concept_info["aliases"]
        for fname, finfo in files_info.items():
            for var_name, vmeta in finfo["variables"].items():
                # Match by variable name or alias
                if var_name.lower() in aliases:
                    concept_mapping[concept_id] = {
                        "actual_name": var_name,
                        "file_name": fname,
                        "file_path": finfo["path"],
                        "units": vmeta["units"],
                        "long_name": vmeta["long_name"],
                    }
                    found = True
                    break
                # Match by long_name keywords
                long_lower = vmeta["long_name"].lower()
                if concept_id == "swh" and ("significant height" in long_lower or "significant wave" in long_lower):
                    concept_mapping[concept_id] = {
                        "actual_name": var_name,
                        "file_name": fname,
                        "file_path": finfo["path"],
                        "units": vmeta["units"],
                        "long_name": vmeta["long_name"],
                    }
                    found = True
                    break
            if found:
                break

    return {
        "files": files_info,
        "variable_to_files": var_to_files,
        "overlapping_variables": overlapping,
        "concept_mapping": concept_mapping,
    }
