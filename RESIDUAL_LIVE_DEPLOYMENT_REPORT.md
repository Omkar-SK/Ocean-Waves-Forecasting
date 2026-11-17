# RESIDUAL DEEP LEARNING LIVE DEPLOYMENT & VALIDATION REPORT
**OceanCast AI Marine Forecasting Platform**  
**Lead Systems Architect & ML Deployment Engineer**  
**Date:** 2026-11-16  
**Operational Status:** Live Ingest & Inference Active | **Geographic Scope:** Mumbai Offshore (Reference Station)

---

## 1. Executive Summary

Following the empirical completion of the OceanCast AI residual-learning experiment, the candidate best-performing residual deep learning models have been integrated into the live forecasting subsystem, backend API endpoints, and operational frontend interface.

The residual formulation changes the neural network prediction target from absolute Significant Wave Height ($\text{SWH}(t+H)$) to the incremental wave height change ($\Delta\text{SWH} = \text{SWH}(t+H) - \text{SWH}(t)$). The absolute forecast is reconstructed at inference time by summing the latest verified in-situ/reanalysis observation ($\text{SWH}(t)$) and the predicted delta ($\Delta\text{SWH}$).

### Candidate Model Portfolio (Held-Out Test Benchmarks)

| Horizon | Candidate Architecture | Checkpoint File | Target | RMSE Skill vs Persistence | MAE Skill vs Persistence | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **+1h** | **Residual WaveLSTM** | `best_lstm_residual_h1.pt` | $\Delta\text{SWH}$ | **+29.76%** | **+45.96%** | `reference_validated` |
| **+6h** | **Residual WaveANN** | `best_ann_residual_h6.pt` | $\Delta\text{SWH}$ | **+18.85%** | **+24.56%** | `reference_validated` |
| **+24h** | **Residual WaveANN** | `best_ann_residual_h24.pt` | $\Delta\text{SWH}$ | **+11.09%** | **+11.34%** | `reference_validated` |

> [!IMPORTANT]
> **Geographic Validation Boundary:**  
> These skill scores are derived strictly from the Mumbai Offshore (18.50°N, 72.50°E) held-out test evaluation ($N=19,431$–$19,454$ hourly steps). These accuracy claims **must not be generalized** to other coastal locations without independent regional calibration and buoy ground truth. All non-reference coastal nodes are explicitly designated as `experimental_unvalidated`.

---

## 2. Central Model Registry

A unified, centralized model registry has been established at [`src/models/registry.py`](file:///d:/OMKAR/VITP/SEM%205/DL/CP/src/models/registry.py).

The registry tracks **18 models** (9 original absolute models and 9 residual models), ensuring programmatic model resolution without hardcoded filenames.

### Registry Entry Schema

Every registered model specifies:
- `model_id`: Canonical unique identifier (e.g. `residual_lstm_h1`, `residual_ann_h6`)
- `architecture`: `lstm` | `ann` | `cnn1d`
- `display_name`: Human-readable label (e.g. `Residual WaveLSTM`, `Residual WaveANN`)
- `forecast_mode`: `ForecastMode.RESIDUAL` or `ForecastMode.ABSOLUTE`
- `horizon_hours`: Lead time (1, 6, 24)
- `checkpoint_path`: Absolute path to PyTorch `.pt` file
- `metadata_path`: Absolute path to companion `_metadata.json` file
- `input_scaler_path`: `models/scaler.pkl` (shared training feature scaler)
- `residual_scaler_path`: `models/residual/residual_scaler_h{H}.pkl` (horizon-specific ΔSWH scaler)
- `target_type`: `delta_swh` (residual) or `swh` (absolute)
- `reconstruction_formula`: `SWH(t+H) = SWH(t) + Predicted_ΔSWH`
- `model_version`: e.g. `2.0.0-residual` vs `1.0.0-absolute`
- `validation_status`: Dynamically evaluated via `get_validation_status(location_id)`

### Public Registry API
A new inspectable endpoint `GET /api/forecast/registry` provides real-time readiness status for all 18 models:
```json
{
  "total_models": 18,
  "ready_models": 18,
  "all_ready": true,
  "candidate_residual_models": {
    "1": { "model_id": "residual_lstm_h1", "rmse_skill_score_vs_persistence": 29.76 },
    "6": { "model_id": "residual_ann_h6", "rmse_skill_score_vs_persistence": 18.85 },
    "24": { "model_id": "residual_ann_h24", "rmse_skill_score_vs_persistence": 11.09 }
  }
}
```

---

## 3. Residual Reconstruction Pipeline

The operational inference engine in [`backend/services/model_service.py`](file:///d:/OMKAR/VITP/SEM%205/DL/CP/backend/services/model_service.py) and [`src/data/live_service.py`](file:///d:/OMKAR/VITP/SEM%205/DL/CP/src/data/live_service.py) enforces the following execution sequence:

```mermaid
flowchart TD
    A["Raw Telemetry (Open-Meteo ECMWF / ERA5 Demo)"] --> B["Input Validation (24h continuous, 0 gaps, 0 NaNs)"]
    B --> C["Feature Preprocessing (13 cyclical/physical variables)"]
    C --> D["Feature Scaling (models/scaler.pkl — NEVER refitted)"]
    D --> E["Tensor Shape Verification: [1, 24, 13]"]
    E --> F["Forward Pass via Selected Candidate Model"]
    F --> G["Output: Scaled ΔSWH"]
    G --> H["Inverse Residual Scaling (models/residual/residual_scaler_h{H}.pkl)"]
    H --> I["Physical ΔSWH in meters"]
    I --> J["Extract Current Observed SWH(t) at latest anchor"]
    J --> K["Reconstruct: SWH(t+H) = SWH(t) + ΔSWH"]
    K --> L["Physical Sanity Enforcement: finite, non-negative, ocean ceiling"]
    L --> M["Response Delivery & Immutable Audit Ledger"]
```

### Critical Architectural Safeguards
1. **Scaler Isolation:**
   - Input features are normalized **strictly** using `models/scaler.pkl` (fitted on 70% training split).
   - The residual target scaler (`residual_scaler_h{H}.pkl`) is **never** applied to input features.
   - Per-horizon residual scalers are strictly isolated:
     - $+1\text{h}: \sigma = 0.0241\text{ m}, \mu \approx 0.000\text{ m}$
     - $+6\text{h}: \sigma = 0.1003\text{ m}, \mu \approx 0.000\text{ m}$
     - $+24\text{h}: \sigma = 0.2114\text{ m}, \mu \approx 0.000\text{ m}$
     Cross-horizon scaler leakage is prevented by explicit path lookups.
2. **Zero-Residual Invariance:**
   - If a model predicts $\Delta\text{SWH} = 0$, the reconstructed forecast identically equals the persistence baseline: $\text{SWH}(t+H) \equiv \text{SWH}(t)$.
3. **Physical Non-Negativity:**
   - Wave height cannot be physically negative: $\text{SWH}_{\text{forecast}} = \max(0.0, \text{SWH}_{\text{forecast}})$.
   - Values exceeding $25.0\text{ m}$ are bounded to realistic marine physical ceilings.

---

## 4. Preservation of Existing Functionality

- **Absolute Models Retained:** All 9 original checkpoints remain fully operational and accessible.
- **Dual-Mode Inference Option:** Callers can toggle between paradigms:
  - `POST /api/forecast` with `{"forecast_mode": "residual"}` (default) or `{"forecast_mode": "absolute"}`.
  - `GET /api/forecast/live?forecast_mode=residual` (default) or `?forecast_mode=absolute`.
- **Persistence Baseline Unchanged:** Baseline persistence continues to report observed $\text{SWH}(t)$.
- **No Hardcoded Fallbacks:** If a checkpoint file is missing or corrupted, the system reports an explicit HTTP 404/500 error or logged fallback with transparent degradation metadata.

---

## 5. API Response Schema Enhancements

The forecast response schemas ([`backend/schemas/forecast.py`](file:///d:/OMKAR/VITP/SEM%205/DL/CP/backend/schemas/forecast.py)) have been updated with complete residual transparency:

```json
{
  "anchor_time_utc": "2026-10-25T12:00:00Z",
  "target_time_utc": "2026-10-25T18:00:00Z",
  "forecast_horizon_hours": 6,
  "persistence_swh_m": 1.2500,
  "current_swh": 1.2500,
  "current_swh_m": 1.2500,
  "predicted_swh": 1.2785,
  "predicted_residual": 0.0285,
  "forecast_mode": "residual",
  "selected_model_architecture": "Residual WaveANN",
  "model_version": "2.0.0-residual",
  "input_timestamp": "2026-10-25T12:00:00Z",
  "data_source": "Offline Demo Fallback: ERA5 Reanalysis Archive",
  "data_freshness": "Calibration Archive",
  "data_freshness_minutes": 5,
  "forecast_generation_timestamp": "2026-11-16T12:55:00Z",
  "model_validation_status": "reference_validated",
  "is_reference": true,
  "location_id": "mumbai_offshore",
  "predictions": {
    "ann": {
      "model_name": "Residual WaveANN",
      "model_type": "ANN",
      "checkpoint_file": "best_ann_residual_h6.pt",
      "forecast_swh_m": 1.2785,
      "predicted_residual_m": 0.0285,
      "current_swh_m": 1.2500,
      "delta_vs_persistence_m": 0.0285,
      "rmse_skill_score": 18.85,
      "is_candidate_model": true,
      "reconstruction_formula": "SWH(t+6) = SWH(t) + Predicted_ΔSWH",
      "forecast_mode": "residual",
      "model_version": "2.0.0-residual",
      "validation_status": "reference_validated",
      "sea_state_desc": "Smooth",
      "sea_state_color": "#22c55e"
    }
  }
}
```

Full backward compatibility is maintained: legacy fields (`forecast_swh_m`, `delta_vs_persistence_m`, etc.) continue to be populated and valid.

---

## 6. Frontend Live Forecast Interface Upgrades

The live forecasting dashboard ([`frontend/src/pages/LiveForecastPage.tsx`](file:///d:/OMKAR/VITP/SEM%205/DL/CP/frontend/src/pages/LiveForecastPage.tsx)) now features:

1. **Horizon Quick-Selection Cards (+1h, +6h, +24h):**
   - Direct toggling across all three supported lead times.
   - Highlights the designated candidate architecture and held-out test skill score (+29.76% for +1h, +18.85% for +6h, +11.09% for +24h).
2. **Clear Observation vs Prediction Demarcation:**
   - **Current Observed SWH ($SWH(t)$):** Displayed with clock telemetry icon.
   - **Predicted Wave-Height Change ($\Delta\text{SWH}$):** Displayed with trend indicators ($\pm\text{meters}$) and color coding.
   - **Reconstruction Formula Badge:** Explains $\text{SWH}(t+H) = \text{SWH}(t) + \Delta\text{SWH}$ directly on the forecast card.
3. **Forecasting Paradigm Switcher:**
   - Pill toggle allowing instantaneous switching between **ΔSWH Residual** and **Absolute** models.
4. **Data Freshness & Regional Status Indicators:**
   - Real-time data freshness pill (`~Xm ago`).
   - Prominent validation badge (`Validated Reference` vs `Unvalidated Experimental Station`).

---

## 7. Numerical Equivalence & Independent Verification

Prior to signing off, a standalone test was executed directly against raw PyTorch checkpoints without utilizing the live service code path. The outputs from `model_service.predict_single` were compared against independent direct execution on ERA5 test windows:

| Horizon | Model | Service Forecast SWH | Independent Direct SWH | Absolute Difference | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **+1h** | Residual WaveLSTM | `0.6716 m` | `0.6716 m` | **0.000030 m** ($< 0.1\text{ mm}$) | **PASSED** |
| **+6h** | Residual WaveANN | `0.6801 m` | `0.6801 m` | **0.000015 m** ($< 0.1\text{ mm}$) | **PASSED** |
| **+24h** | Residual WaveANN | `0.7006 m` | `0.7006 m` | **0.000035 m** ($< 0.1\text{ mm}$) | **PASSED** |

---

## 8. Original Checkpoints Preservation Audit

All 9 original checkpoints in `models/` were audited for file presence, byte size, and git immutability:

| Checkpoint File | Byte Size | State |
| :--- | :--- | :--- |
| `models/best_ann.pt` | 632,551 bytes | **Unmodified (100% Intact)** |
| `models/best_ann_h6.pt` | 632,764 bytes | **Unmodified (100% Intact)** |
| `models/best_ann_h24.pt` | 632,835 bytes | **Unmodified (100% Intact)** |
| `models/best_lstm.pt` | 682,623 bytes | **Unmodified (100% Intact)** |
| `models/best_lstm_h6.pt` | 682,785 bytes | **Unmodified (100% Intact)** |
| `models/best_lstm_h24.pt` | 682,839 bytes | **Unmodified (100% Intact)** |
| `models/best_cnn1d.pt` | 942,197 bytes | **Unmodified (100% Intact)** |
| `models/best_cnn1d_h6.pt` | 942,377 bytes | **Unmodified (100% Intact)** |
| `models/best_cnn1d_h24.pt` | 942,437 bytes | **Unmodified (100% Intact)** |

`git diff models/` reports zero diffs.

---

## 9. Automated Test Suite Execution Summary

### Backend PyTest Suite
```
tests/test_api.py ......................... [PASSED]
tests/test_coastal_risk_map.py ............ [PASSED]
tests/test_data_validation.py ............. [PASSED]
tests/test_evaluation.py .................. [PASSED]
tests/test_forecasting_pipeline.py ........ [PASSED]
tests/test_live_forecasting.py ............ [PASSED]
tests/test_model_drift_monitoring.py ...... [PASSED]
tests/test_multi_location_foundation.py ... [PASSED]
tests/test_preprocessing.py ............... [PASSED]
tests/test_residual_live_integration.py ... [PASSED]
tests/test_risk_and_alerts.py ............. [PASSED]
tests/test_simulation.py .................. [PASSED]

====================== 125 passed, 6 warnings in 20.79s =======================
```

### Dedicated Residual Integration Tests ([`tests/test_residual_live_integration.py`](file:///d:/OMKAR/VITP/SEM%205/DL/CP/tests/test_residual_live_integration.py))
- `test_residual_checkpoints_loading`: **PASSED**
- `test_residual_scalers_loading_and_isolation`: **PASSED**
- `test_input_preprocessing_and_tensor_dimensions`: **PASSED**
- `test_residual_inverse_transformation`: **PASSED**
- `test_forecast_reconstruction_formula`: **PASSED**
- `test_zero_residual_equals_persistence`: **PASSED**
- `test_nan_rejection_in_input`: **PASSED**
- `test_missing_feature_rejection`: **PASSED**
- `test_invalid_sequence_rejection`: **PASSED**
- `test_candidate_selection_all_three_horizons`: **PASSED**
- `test_fastapi_live_api_residual_endpoints`: **PASSED**
- `test_concurrent_multi_threaded_inference`: **PASSED**
- `test_geographic_validation_status_isolation`: **PASSED**

### Frontend Vitest Suite
```
Test Files  9 passed (9)
Tests       46 passed (46)
Duration    11.14s
```
Includes expanded test coverage for horizon cards (+1h, +6h, +24h), observed vs predicted delta separation, data freshness rendering, and forecast mode switching.

---

## 10. Operational Summary & Recommendations

1. **Deployment Success Verified:** The validated residual forecasting engine is active and serving inference across all horizons (+1h, +6h, +24h) with verified mathematical equivalence to raw PyTorch execution.
2. **Geographic Gating Active:** Non-reference stations (Goa, Mangalore, Kochi, Kanyakumari, Chennai, Visakhapatnam, Paradip) are strictly demarcated as `experimental_unvalidated`.
3. **Repository State:** Branch `main` has changes unstaged as instructed. No git commit or push has been performed (`DONT COMMIT UNLESS I TELL`).
