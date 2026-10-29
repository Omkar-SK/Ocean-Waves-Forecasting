# OceanCast AI: Real-Time Ocean Intelligence & Coastal Early-Warning System

> **Primary Application:** Complete **React + Vite + TypeScript** web application (`frontend/`).  
> **Legacy / Fallback Interface:** Streamlit dashboard (`app.py`) is preserved for archival reference only and is **no longer required** for primary operational use.

OceanCast AI is an operational deep learning marine forecasting and coastal intelligence platform comparing **Artificial Neural Networks (ANN)**, **Long Short-Term Memory Networks (LSTM)**, and **1D Convolutional Neural Networks (1D-CNN)** for predicting significant wave height ($H_s$) using high-resolution ECMWF ERA5 reanalysis and live ECMWF WAM marine telemetry.

The platform provides multi-horizon forecasts ($+1\text{h}, +6\text{h}, +24\text{h}$), interactive geographic coastal risk visualization across 8 Indian coastal candidate nodes, persistent early-warning alert lifecycle management, prospective forecast accuracy tracking, and automated model drift monitoring.

---

## 🚀 Quick Start: Launching the Application

### 1. Prerequisites

- **Python**: 3.10+ (tested on Python 3.11)
- **Node.js**: 18+ (tested with npm 10+)

```bash
# In project root: install backend Python dependencies
pip install -r requirements.txt

# In frontend directory: install React dependencies
cd frontend
npm install
cd ..
```

---

### 2. Environment Variables Configuration

Create local environment files from the provided templates:

**Backend (`.env` in root directory):**
```bash
# Copy example configuration
cp .env.example .env
```
Key variables:
- `CORS_ORIGINS`: Allowed client origins, e.g., `http://localhost:5173,http://127.0.0.1:5173`
- `OFFSHORE_LAT`: Reference latitude (`18.50`)
- `OFFSHORE_LON`: Reference longitude (`72.50`)
- `DB_PATH`: SQLite database file path (`ocean_intelligence.db`)

**Frontend (`frontend/.env`):**
```bash
# Copy example frontend configuration
cd frontend
cp .env.example .env
cd ..
```
Key variables:
- `VITE_API_BASE_URL`: Backend API endpoint base (defaults to empty string `""` to use Vite proxy `/api`, or `http://127.0.0.1:8000/api`)

---

### 3. Dual-Terminal Local Launch

The application runs as a modern decoupled web architecture:

#### Terminal 1 — FastAPI Backend (Port 8000)
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

#### Terminal 2 — React + Vite Frontend (Port 5173)
```bash
cd frontend
npm run dev
```
- Primary User Interface: **`http://localhost:5173`**

---

### 4. Running Automated Tests

#### Backend Test Suite (Python Pytest)
```bash
pytest -v
```
*Current test status:* **104 passed** (covering data validation, preprocessing, PyTorch models, forecasting pipelines, coastal map data, alert lifecycles, and monitoring endpoints).

#### Frontend Test Suite (Vitest & React Testing Library)
```bash
cd frontend
npm test
```
*Current test status:* **18 passed** across 6 test suites (testing application routing, centralized API client, live forecasting, interactive Leaflet risk map, coastal alerts acknowledgement, and model drift telemetry).

#### Production Frontend Bundle Build
```bash
cd frontend
npm run build
```
*Build output:* Clean production bundle in `frontend/dist/`.

---

### 5. Archival / Secondary Streamlit Interface (Optional)

The legacy Streamlit dashboard is preserved for archival reference:
```bash
streamlit run app.py
```
> **Notice:** Streamlit is optional and not required to run or use OceanCast AI. The React + Vite application provides complete feature parity and expanded capabilities.

---

## 🖥️ React + Vite Application Pages (9 Core Sections)

| Page | Route | Description | Backend APIs |
| :--- | :--- | :--- | :--- |
| **1. Overview Dashboard** | `/` | Operational mission control: real-time telemetry, $+1\text{h}/+6\text{h}/+24\text{h}$ multi-horizon cards, dual-section trajectory chart, active alert counts, current coastal risk badge, and system readiness. | `/api/health`<br>`/api/forecast/live`<br>`/api/alerts/summary`<br>`/api/coastal/metrics` |
| **2. Live Wave Forecasting** | `/live` | Multi-location forecasting: coastal station selector (8 nodes), experimental validation notices, 3-way benchmark (Deep Learning vs ECMWF NWP vs Naive Persistence), and 24h sequence time-series. | `/api/coastal/locations`<br>`/api/forecast/status`<br>`/api/forecast/live` |
| **3. Coastal Risk Heatmap** | `/map` | Real interactive Leaflet geographic map: 8 candidate coastal nodes rendered with risk categories (`GREEN`, `YELLOW`, `ORANGE`, `RED`), node details drawer, horizon toggle, alert filters, and threshold legend. | `/api/coastal/map-data`<br>`/api/coastal/thresholds` |
| **4. Coastal Alerts & Early Warnings** | `/alerts` | Full early-warning alert centre: active alert cards, severity & location filters, operator acknowledgement with persistent notes, lifecycle history modal, and manual re-evaluation trigger. | `/api/alerts`<br>`/api/alerts/summary`<br>`/api/alerts/{id}/acknowledge`<br>`/api/alerts/evaluate`<br>`/api/alerts/thresholds` |
| **5. Coastal Monitoring Nodes** | `/locations` | Multi-node ingestion monitor: comprehensive 8-node telemetry grid, feed freshness (`FRESH` vs `STALE`), observed wave heights, and clear distinction between Mumbai reference and experimental transfers. | `/api/coastal/locations`<br>`/api/coastal/metrics`<br>`/api/coastal/map-data` |
| **6. Forecast Accuracy & Drift** | `/monitoring` | Operational ML monitoring: prospective forecast ledger, matched observation pairs, coverage %, verified MAE/RMSE/bias, persistence comparisons, Recharts PSI feature drift bar chart, and degradation tracker. | `/api/monitoring/summary`<br>`/api/monitoring/accuracy`<br>`/api/monitoring/drift`<br>`/api/monitoring/events`<br>`/api/monitoring/ledger`<br>`/api/monitoring/backtest` |
| **7. Historical Test Analysis** | `/historical` | Ground-truth analysis on $19,455$ untouched test records ($2024\text{--}2026$): Monsoon/Calm presets, 1:1 actual vs predicted scatter plot, residual error histogram, and raw CSV dataset export. | `/api/historical/summary`<br>`/api/historical/predictions`<br>`/api/historical/scatter`<br>`/api/historical/residuals`<br>`/api/historical/export` |
| **8. Model Comparison** | `/comparison` | Ground-truth benchmarking: grouped bar charts comparing ANN, LSTM, 1D-CNN, and Persistence across $+1\text{h}, +6\text{h}, +24\text{h}$, skill score analysis, and exact academic metrics table. | `/api/metrics/multi-horizon`<br>`/api/metrics` |
| **9. System Status & Docs** | `/about` | Operational health status of FastAPI, Open-Meteo, PyTorch checkpoints, honest scientific disclosures (absence of live waverider buoys, no spatial interpolation), and canonical 13-feature architecture. | `/api/health`<br>`/api/metadata`<br>`/api/forecast/status` |

---

## 🏛️ System Architecture

```text
OceanCast AI Platform
├── backend/                                   # FastAPI Web Service & ML Orchestration
│   ├── main.py                               # App entry point, CORS, lifespan, routes
│   ├── config.py                             # Paths, offshore coordinates, feature defs
│   ├── schemas/                              # Pydantic v2 validation models
│   │   ├── alerts.py                         # Alert item, summary, acknowledge schemas
│   │   ├── coastal.py                        # Coastal node map, metrics, threshold schemas
│   │   ├── forecast.py                       # Forecast request & response schemas
│   │   ├── health.py                         # Health & metadata schemas
│   │   ├── historical.py                     # Historical query & residual schemas
│   │   ├── metrics.py                        # Model metrics schemas
│   │   └── monitoring.py                     # Accuracy, drift & ledger schemas
│   ├── services/                             # Business logic & ML inference
│   │   ├── model_service.py                  # PyTorch model caching & inference
│   │   ├── live_service.py                   # Open-Meteo REST client & consecutive validation
│   │   ├── metrics_service.py                # Multi-horizon comparison parser
│   │   └── historical_service.py             # Downsampled time series, scatter, residuals
│   └── routers/                              # REST API endpoint handlers
│       ├── alerts.py                         # /api/alerts, /acknowledge, /evaluate, /thresholds
│       ├── coastal.py                        # /api/coastal/map-data, /metrics, /thresholds
│       ├── forecast.py                       # /api/forecast/status, /api/forecast/live
│       ├── health.py                         # /api/health, /api/metadata
│       ├── historical.py                     # /api/historical/predictions, /export
│       ├── metrics.py                        # /api/metrics/multi-horizon, /api/metrics
│       └── monitoring.py                     # /api/monitoring/summary, /accuracy, /drift, /ledger
│
├── frontend/                                  # React + Vite + TypeScript Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/                       # Sidebar, Header, Layout
│   │   │   └── ui/                           # Card, Badge, Button, StatusIndicator, Skeleton
│   │   ├── pages/                            # 9 Complete Migrated Dashboard Pages
│   │   │   ├── OverviewPage.tsx
│   │   │   ├── LiveForecastPage.tsx
│   │   │   ├── CoastalRiskMapPage.tsx        # Leaflet interactive map
│   │   │   ├── AlertsPage.tsx                # Early-warning alert management
│   │   │   ├── CoastalNodesPage.tsx          # Multi-location grid
│   │   │   ├── ForecastAccuracyDriftPage.tsx # Accuracy & PSI drift
│   │   │   ├── HistoricalAnalysisPage.tsx    # 19.4k test drill-down
│   │   │   ├── ModelComparisonPage.tsx       # Grouped bar charts & skill tables
│   │   │   └── AboutPage.tsx                 # System status & methodology
│   │   ├── services/api.ts                   # Centralized typed API client
│   │   ├── types/index.ts                    # TypeScript interfaces matching backend models
│   │   ├── lib/utils.ts                      # Utilities, WMO sea state classifier
│   │   ├── setupTests.ts                     # Vitest test setup (Leaflet & ResizeObserver mocks)
│   │   ├── App.tsx                           # Route tree with React Router
│   │   └── index.css                         # Dark ocean theme design system (Tailwind v4)
│   ├── __tests__/                            # 6 Vitest test suites (18 tests)
│   ├── package.json
│   ├── vite.config.ts                        # Proxy to backend /api
│   └── tsconfig.app.json
│
├── src/                                       # Core Python ML & Intelligence Library
│   ├── intelligence/                         # Coastal intelligence services
│   │   ├── nodes.py                          # 8 Coastal candidate node definitions
│   │   ├── provider.py                       # Marine data-provider abstraction
│   │   ├── ingestion.py                      # Multi-location ingestion pipeline
│   │   ├── risk.py                           # Risk assessment & thresholds
│   │   ├── alerts.py                         # Persistent SQLite alert store & hysteresis
│   │   ├── ledger.py                         # Forecast ledger & observation recorder
│   │   ├── verification.py                   # Prospective observation matching
│   │   └── drift.py                          # Population Stability Index (PSI) drift engine
│   ├── data/live_service.py                  # Live Open-Meteo feature pipeline
│   ├── models/                               # PyTorch model definitions (ANN, LSTM, CNN1D)
│   └── evaluation/                           # Evaluation & metric calculation routines
│
├── models/                                    # 9 PyTorch checkpoints (.pt) and scaler.pkl
├── outputs/                                   # Metrics CSVs, evaluation plots, test predictions
├── ocean_intelligence.db                     # SQLite persistence (alerts, ledger, monitoring)
├── tests/                                     # 104 automated pytest tests
├── app.py                                     # Streamlit interface (archival / legacy fallback)
└── requirements.txt                           # Python dependencies
```

---

## 🔬 Scientific and Machine Learning Integrity

OceanCast AI adheres to strict academic and operational standards:

1. **Preserved PyTorch Checkpoints**: All 9 trained model checkpoints (`WaveANN`, `WaveLSTM`, `WaveCNN1D` across $1\text{h}, 6\text{h}, 24\text{h}$) and `models/scaler.pkl` remain untouched.
2. **Canonical 13-Feature Space**: Exact feature sequence: `[swh, mwp, mwd_sin, mwd_cos, wind_speed, wind_dir_sin, wind_dir_cos, sst, prmsl, month_sin, month_cos, hour_sin, hour_cos]`.
3. **Rigorous Input Tensors**: Consistent `(batch, 24, 13)` 24-hour sequence tensor requirements.
4. **Honest Observation Attribution**:
   - Absence of live waverider buoys (INCOIS unconfigured) is explicitly disclosed.
   - Retrospective ERA5 analysis is clearly separated from prospective observation verification.
   - Evidence states: `VERIFIED`, `LIMITED_EVIDENCE`, `AWAITING_OBSERVATIONS`, `VERIFICATION_UNAVAILABLE`, `RETROSPECTIVE_REANALYSIS`.
5. **No Spatial Interpolation**: Explicit prohibition against unsupported spatial contouring between sparse coastal nodes.
6. **Geographic Validation Status**: Mumbai reference location ($18.50^\circ\text{N}, 72.50^\circ\text{E}$) is clearly marked as validated; remaining 7 nodes are explicitly designated as unvalidated experimental regional transfers.

---

## 📊 Ground-Truth Multi-Horizon Benchmark

Evaluated on $19,455$ consecutive untouched test hours ($2024\text{-}07\text{-}29\text{ to }2026\text{-}10\text{-}17$):

| Horizon | Model | Test RMSE (m) | Test MAE (m) | $R^2$ Score | RMSE Skill vs Persistence | MAE Skill vs Persistence | Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | **Naive Persistence** | **0.0213** | **0.0131** | **0.9991** | *Baseline (0.0%)* | *Baseline (0.0%)* | **Best Overall** |
| +1h | WaveLSTM (Recurrent) | 0.0271 | 0.0182 | 0.9986 | -27.20% | -38.55% | Best Neural Net |
| +1h | WaveANN (MLP) | 0.0465 | 0.0349 | 0.9958 | -118.81% | -166.39% | — |
| +1h | WaveCNN1D (Temporal) | 0.0587 | 0.0426 | 0.9934 | -176.08% | -224.71% | — |
| **+6h** | **WaveLSTM (Recurrent)** | **0.0769** | **0.0533** | **0.9886** | **+13.88%** | **+14.51%** | **Best Overall** |
| +6h | WaveANN (MLP) | 0.0815 | 0.0574 | 0.9872 | +8.77% | +7.92% | Beats Baseline |
| +6h | Naive Persistence | 0.0893 | 0.0624 | 0.9846 | *Baseline (0.0%)* | *Baseline (0.0%)* | — |
| +6h | WaveCNN1D (Temporal) | 0.0942 | 0.0679 | 0.9829 | -5.43% | -8.81% | — |
| **+24h** | **Naive Persistence** | **0.1866** | 0.1286 | **0.9330** | *Baseline (0.0%)* | *Baseline (0.0%)* | **Lowest RMSE** |
| +24h | WaveCNN1D (Temporal) | 0.1901 | 0.1362 | 0.9305 | -1.90% | -5.88% | — |
| +24h | **WaveANN (MLP)** | 0.1920 | **0.1225** | 0.9291 | -2.89% | **+4.71%** | **Lowest MAE** |
| +24h | WaveLSTM (Recurrent) | 0.1979 | 0.1229 | 0.9247 | -6.06% | **+4.45%** | Competitive MAE |

---

## 🛠️ Complete REST API Reference

| Method | Endpoint | Group | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | System | Health readiness, PyTorch checkpoints, and component health |
| `GET` | `/api/metadata` | System | Academic metadata, 13 canonical features, sequence length |
| `GET` | `/api/forecast/status` | Forecasting | Ingestion status, observation timestamps, checkpoint readiness |
| `GET` | `/api/forecast/live` | Forecasting | Multi-location forecast generation ($+1\text{h}, +6\text{h}, +24\text{h}$) |
| `POST` | `/api/forecast` | Forecasting | Direct tensor inference with custom 24-step input array |
| `GET` | `/api/coastal/locations` | Coastal | 8 coastal candidate nodes with coordinates and status |
| `GET` | `/api/coastal/map-data` | Coastal | Geographic map items, risk classifications, multi-horizon forecasts |
| `GET` | `/api/coastal/metrics` | Coastal | Node counts, high risk node counts, degraded feed tallies |
| `GET` | `/api/coastal/thresholds` | Coastal | Engineering thresholds ($H_s$ caution, high risk, critical) |
| `GET` | `/api/alerts` | Alerts | Filtered list of persistent early-warning coastal alerts |
| `GET` | `/api/alerts/summary` | Alerts | Summary counts: total active, critical, high, caution, resolved |
| `POST` | `/api/alerts/{id}/acknowledge` | Alerts | Operator acknowledgement with timestamp and notes |
| `GET` | `/api/alerts/{id}/history` | Alerts | Lifecycle audit trail for a specific alert |
| `POST` | `/api/alerts/evaluate` | Alerts | Force evaluation of multi-location forecasts against thresholds |
| `GET` | `/api/alerts/thresholds` | Alerts | Configured early-warning threshold settings |
| `GET` | `/api/monitoring/summary` | Monitoring | Ledger counts, verification coverage, buoy availability |
| `GET` | `/api/monitoring/accuracy` | Monitoring | Prospective accuracy metrics (MAE, RMSE, bias, skill score) |
| `GET` | `/api/monitoring/drift` | Monitoring | PSI feature drift report across all 13 canonical variables |
| `GET` | `/api/monitoring/events` | Monitoring | Audit log of data degradation and drift events |
| `GET` | `/api/monitoring/ledger` | Monitoring | Recent forecast issuance records from SQLite ledger |
| `POST` | `/api/monitoring/backtest` | Monitoring | Populate ledger and observations from historical ERA5 records |
| `GET` | `/api/metrics/multi-horizon` | Metrics | Ground-truth multi-horizon comparison table |
| `GET` | `/api/metrics` | Metrics | Filtered metrics for a specific horizon ($1\text{h}, 6\text{h}, 24\text{h}$) |
| `GET` | `/api/historical/summary` | Historical | Test split statistics (record counts, horizons, date ranges) |
| `GET` | `/api/historical/predictions`| Historical | Downsampled actual vs predicted time series for charting |
| `GET` | `/api/historical/scatter` | Historical | Downsampled actual vs predicted scatter plot points |
| `GET` | `/api/historical/residuals` | Historical | Residual error distribution histogram bins and statistics |
| `GET` | `/api/historical/export` | Historical | Raw test set predictions CSV download |

---

## 💡 Troubleshooting

- **FastAPI Backend fails to start**: Ensure Python dependencies are installed (`pip install -r requirements.txt`). Ensure port 8000 is available.
- **Frontend shows "Backend Offline"**: Verify the backend is running at `http://127.0.0.1:8000`. The Vite development server automatically proxies `/api` requests to port 8000.
- **Port Conflict**: If port 8000 is in use, start Uvicorn on another port (e.g. `8080`) and configure `VITE_API_BASE_URL=http://127.0.0.1:8080/api` in `frontend/.env`.
