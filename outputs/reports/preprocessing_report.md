# ERA5 Ocean Wave & Atmospheric Time Series Preprocessing Report

## 1. Selected Offshore Grid Point
- **Requested Coordinates**: 18.50° N, 72.50° E
- **Selected Coordinates**: 18.50° N, 72.50° E
- **Exact Grid Intersection**: Yes
- **Water Verification**: Confirmed (high certainty)
- **Audit Note**: Point (18.50°N, 72.50°E) has 100% valid wave observations (129,696 hours, range: [0.36m, 7.08m], mean: 1.25m). Zero NaNs confirm this is an offshore ocean cell.

## 2. Temporal Span & Data Continuity
- **Start Time**: 2012-01-01 00:00:00
- **End Time**: 2026-10-17 23:00:00
- **Total Synchronized Rows**: 129,696
- **Sampling Frequency**: 1 hour (hourly continuous)
- **Duplicate Timestamps**: 0
- **Missing Timestamps**: 0
- **Maximum Gap Interval**: 1.0 hour(s)

## 3. Data Variables & Engineered Features
| Variable | Role | Units | Description |
| :--- | :--- | :--- | :--- |
| `swh` | Primary Prediction Target | m | Significant height of combined wind waves and swell |
| `mwp` | Input Feature | s | Mean wave period |
| `mwd_deg` | Raw Wave Direction | Degree true [0, 360] | Wave propagation direction |
| `mwd_sin` | Circular Feature | [-1.0, 1.0] | sin(deg2rad(mwd)) |
| `mwd_cos` | Circular Feature | [-1.0, 1.0] | cos(deg2rad(mwd)) |
| `u10` | Input Feature | m/s | 10m zonal wind component |
| `v10` | Input Feature | m/s | 10m meridional wind component |
| `wind_speed` | Derived Feature | m/s | sqrt(u10^2 + v10^2) |
| `wind_direction_deg` | Derived Feature | Degree true [0, 360] | Meteorological wind direction |
| `wind_dir_sin` | Circular Feature | [-1.0, 1.0] | sin(deg2rad(wind_dir)) |
| `wind_dir_cos` | Circular Feature | [-1.0, 1.0] | cos(deg2rad(wind_dir)) |
| `hour_sin / hour_cos` | Temporal Feature | [-1.0, 1.0] | Diurnal cycle 24h sine/cosine |
| `month_sin / month_cos` | Temporal Feature | [-1.0, 1.0] | Annual cycle 12m sine/cosine |

## 4. Summary Statistics (Clean Processed Dataset)
| Variable | Min | Mean | Max | Std |
| :--- | :--- | :--- | :--- | :--- |
| `swh` | 0.3605 | 1.2485 | 7.0831 | 0.7321 |
| `mwp` | 3.9557 | 7.3169 | 13.5536 | 1.3786 |
| `mwd` | 0.0112 | 257.5230 | 360.0004 | 43.4453 |
| `u10` | -19.7562 | 2.4379 | 15.3282 | 3.2993 |
| `v10` | -18.9973 | -1.4722 | 24.1080 | 3.7910 |

## 5. Artifacts Created
- `D:\OMKAR\VITP\SEM 5\DL\CP\data\processed\era5_wave_offshore_timeseries.parquet`
- `D:\OMKAR\VITP\SEM 5\DL\CP\data\processed\era5_wave_offshore_timeseries.csv`
