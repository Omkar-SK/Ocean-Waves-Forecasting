# OceanCast AI: Deep Learning Ocean Wave Forecasting Dashboard
### Academic Project: Ocean Wave Forecasting Using Deep Learning (ANN vs LSTM vs 1D-CNN)

An academic and operational deep learning system comparing **Artificial Neural Networks (ANN)**, **Long Short-Term Memory Networks (LSTM)**, and **1D Convolutional Neural Networks (1D-CNN)** for predicting significant wave height ($H_s$) using high-resolution ECMWF ERA5 reanalysis and operational ECMWF WAM marine telemetry.

The project features a **React + Vite + TypeScript** frontend with Tailwind CSS and Recharts, backed by a high-throughput **FastAPI** Python service serving PyTorch checkpoints.

---

## 🌊 Quick Start: Launching the Application

### 1. Prerequisites & Environment Setup

#### Python Backend
Ensure Python 3.10+ is installed:
```bash
# In the project root (D:\OMKAR\VITP\SEM 5\DL\CP)
pip install -r requirements.txt
```

#### Frontend Dependencies
Ensure Node.js 18+ is installed:
```bash
cd frontend
npm install
cd ..
```

---

### 2. Running the Full-Stack Application

Start the **FastAPI Backend** and the **React + Vite Frontend**:

#### Terminal 1 — FastAPI Backend (Port 8000)
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs (Swagger): `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/health`

#### Terminal 2 — React + Vite Frontend (Port 5173)
```bash
cd frontend
npm run dev
```
- Dashboard URL: **`http://localhost:5173`**

---

### 3. Running Automated Tests

Run the full automated test suite (36 ML tests + 12 API integration tests):
```bash
pytest -v
```

Build and validate the frontend production bundle:
```bash
cd frontend
npm run build
```

---

### 4. Optional Streamlit Fallback

The original Streamlit interface is preserved as an optional fallback:
```bash
streamlit run app.py
```

---

## 📂 Project Architecture

```text
CP/
├── backend/                                   # FastAPI Web Service
│   ├── main.py                               # App entry point, CORS, lifespan, routes
│   ├── config.py                             # Paths, offshore coordinates, feature defs
│   ├── schemas/                              # Pydantic v2 validation models
│   │   ├── health.py                         # Health & metadata schemas
│   │   ├── metrics.py                        # Model metrics schemas
│   │   ├── forecast.py                       # Forecast request & response schemas
│   │   └── historical.py                     # Historical query & residual schemas
│   ├── services/                             # Business logic & ML inference
│   │   ├── model_service.py                  # PyTorch model caching & inference
│   │   ├── metrics_service.py                # Multi-horizon comparison parser
│   │   ├── historical_service.py             # Downsampled time series, scatter, residuals
│   │   └── live_service.py                   # Open-Meteo REST client & consecutive validation
│   └── routers/                              # REST API endpoint handlers
│       ├── health.py                         # /api/health, /api/metadata
│       ├── metrics.py                        # /api/metrics/multi-horizon, /api/metrics
│       ├── forecast.py                       # /api/forecast/status, /api/forecast/live
│       └── historical.py                     # /api/historical/predictions, /api/historical/export
│
├── frontend/                                  # React + Vite + TypeScript Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/                       # Sidebar, Header, Layout
│   │   │   └── ui/                           # Card, Badge, Button, StatusIndicator, Skeleton
│   │   ├── pages/
│   │   │   ├── OverviewPage.tsx              # Telemetry, trajectory chart, health status
│   │   │   ├── LiveForecastPage.tsx          # Near-real-time prediction, 3-way benchmark
│   │   │   ├── ModelComparisonPage.tsx       # Grouped bar charts, R² panels, skill analysis
│   │   │   ├── HistoricalAnalysisPage.tsx    # 19.4k test drill-down, 1:1 scatter, residuals, CSV export
│   │   │   └── AboutPage.tsx                 # Methodology, pipeline diagram, 13 features
│   │   ├── services/
│   │   │   └── api.ts                        # Typed API client with error handling
│   │   ├── types/                            # TypeScript interfaces matching backend models
│   │   ├── lib/                              # utils.ts, WMO sea state classifier
│   │   ├── App.tsx                           # React Router route tree
│   │   └── index.css                         # Dark ocean theme design system (Tailwind v4)
│   ├── package.json
│   ├── vite.config.ts                        # Proxy to backend /api
│   └── tsconfig.json
│
├── data/
│   ├── interim/                              # Transformed intermediate arrays
│   ├── processed/                            # Preprocessed datasets (Parquet + CSV)
│   ├── reanalysis-era5-single-levels-timeseries-sfcxafzi82j.nc  # ERA5 wind (u10, v10)
│   └── reanalysis-era5-single-levels-timeseries-wavclee9pvb.nc  # ERA5 waves (swh, mwp, mwd)
├── models/                                   # PyTorch checkpoints (.pt) and scaler.pkl
├── outputs/
│   ├── metrics/                              # multi_horizon_comparison.csv, detailed JSONs
│   ├── plots/                                # Historical evaluation plots
│   └── predictions/                          # test_predictions.csv (19,455 test records)
├── src/                                      # Reusable ML library (data, models, training)
├── tests/                                    # Automated pytest test suite
├── app.py                                    # Streamlit interface (legacy fallback)
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## 🖥️ OceanCast AI Dashboard Pages

### 1. Mission Overview (`/`)
- **Key Telemetry Cards**: Latest significant wave height ($H_s$), reference timestamp, dynamic best model for selected horizon, and active checkpoint count (9/9).
- **Trajectory Chart**: Dual-section time-series showing historical hourly observations transitioning across an **Input Cutoff** reference line into future forecasts for WaveLSTM, WaveANN, WaveCNN1D, and Persistence.
- **Horizon Selector**: Dynamic toggle between $+1\text{h}$, $+6\text{h}$, and $+24\text{h}$ horizons.
- **System Readiness Telemetry**: Verified live indicators for ERA5 data, PyTorch checkpoints, and feature scalers.

### 2. Live Operational Forecast (`/live`)
- **Scientific Honesty**: Explicitly separates retrospective test predictions from operational live forecasts. Includes an **Offline Historical Demo Mode** banner disclosing when inference runs on the verified ERA5 window ending 2026-10-17 23:00 UTC.
- **3-Way Benchmark**: Direct side-by-side comparison between Deep Learning predictions, Physics-based NWP (ECMWF WAM), and Naive Persistence.
- **Operational Controls**: Target horizon ($+1\text{h}, +6\text{h}, +24\text{h}$), architecture selection, and data source fallback toggle.
- **Sea State Intelligence**: WMO Sea State code classification (e.g., Code 3 Slight, Code 4 Moderate) with peak wave period ($T_p$) and 10m wind speed ($U_{10}$).

### 3. Model Benchmarking & Skill Analysis (`/comparison`)
- **Grouped Bar Charts**: Interactive comparisons for RMSE and MAE across all three horizons.
- **Complete Benchmark Table**: Exact reproduction of ground-truth test metrics with RMSE and MAE skill relative to Persistence.
- **Academic Insights**:
  - **$+1\text{h}$ Horizon**: Persistence leads short horizons due to extreme ocean inertia ($\rho_1 = 0.9995$). WaveLSTM is the best neural network ($0.0271\text{ m}$ RMSE).
  - **$+6\text{h}$ Horizon**: WaveLSTM decisively outperforms persistence with **$+13.88\%$ RMSE skill** ($0.0769\text{ m}$) and **$+14.51\%$ MAE skill** ($0.0533\text{ m}$).
  - **$+24\text{h}$ Horizon**: Persistence maintains lowest squared peak error ($0.1866\text{ m}$ RMSE), while WaveANN achieves lowest absolute error (**$+4.71\%$ MAE skill**).

### 4. Historical Test Set Evaluation (`/historical`)
- **19,455 Untouched Test Records**: Rigorous evaluation spanning July 2024 to October 2026.
- **Sea State Regimes**: One-click presets for Monsoon Peak, Winter Calm, and Pre-Monsoon transitions.
- **Interactive Scatter Plot**: Actual vs. Predicted values aligned with a 1:1 diagonal reference line.
- **Residual Distribution**: Histogram of error ($y - \hat{y}$) with mean error and standard deviation.
- **Data Export**: Direct CSV download button streaming evaluation data directly from the backend.

### 5. About & Methodology (`/about`)
- **Academic Context**: Semester 5 Deep Learning Capstone Project (VIT Pune).
- **Interactive Pipeline Diagram**: 6-step architecture from ERA5 ingestion to multi-horizon evaluation.
- **13 Canonical Features**: Clear breakdown of wave, atmospheric, circular trigonometric, and seasonal features.
- **Scientific Caveats**: Discussion of reanalysis smoothing vs. buoy measurements and operational NWP distribution shift.

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

## 🛠️ API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System readiness, loaded models, checkpoint map, and coordinates |
| `GET` | `/api/metadata` | Academic title, target location, 13 features, sequence length (24h) |
| `GET` | `/api/metrics/multi-horizon` | Full multi-horizon benchmark comparison table |
| `GET` | `/api/metrics?horizon={h}` | Filtered metrics for horizon $h \in \{1, 6, 24\}$ |
| `GET` | `/api/forecast/status` | Operational feed status and timestamp validation |
| `GET` | `/api/forecast/live?horizon={h}&model_name={m}` | Near-real-time inference with fallback |
| `POST` | `/api/forecast` | Direct inference with custom 24-step input tensor |
| `GET` | `/api/historical/summary` | Test split statistics (sample count, dates, horizons) |
| `GET` | `/api/historical/predictions` | Downsampled time-series for 60fps charting |
| `GET` | `/api/historical/scatter` | Actual vs. Predicted pairs for 1:1 scatter plot |
| `GET` | `/api/historical/residuals` | Residual error histogram bins and statistics |
| `GET` | `/api/historical/export` | Raw test predictions CSV stream download |

---

## 💡 Troubleshooting

- **Backend fails to start**: Ensure PyTorch and FastAPI dependencies are installed: `pip install -r requirements.txt`. Check that port 8000 is free.
- **Frontend shows "Backend Offline"**: Verify the backend is running at `http://127.0.0.1:8000`. The Vite development proxy forwards all `/api` requests to port 8000.
- **Memory errors during test execution**: Parquet files are large; test routines automatically use sliced views and CSV fallbacks. Run `pytest -v` to verify.
