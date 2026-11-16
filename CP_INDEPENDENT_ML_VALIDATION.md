# OceanCast AI / Coastal Prediction (CP)
# Independent ML Research Audit & Deep Learning Validation Report

**Auditor Role:** Independent Machine Learning Research Auditor & Deep Learning Validation Engineer  
**Date of Independent Validation:** November 16, 2026  
**Audited Repository Branch:** `main` (Commit `2873ced`)  
**Validation Methodology:** Full empirical reproduction from raw source code, PyTorch model checkpoints, processed Parquet datasets, Open-Meteo REST telemetry, and mathematical derivations (Zero reliance on previous audit claims).  
**Core Verdict:** **SCIENTIFICALLY SOUND & GENUINE DL PIPELINE with SPECIFIC EMPIRICAL REVISIONS (Zero Mocking, Zero Data Leakage; 1 Major Metric Discrepancy Corrected).**

---

## Executive Summary of Independent Findings

This report presents an uncompromised, independent scientific evaluation of the deep learning forecasting system within the OceanCast AI / Coastal Prediction project. Every claim, weight tensor, metric, and data pipeline stage was independently reproduced.

### High-Level Audit Classification

| Investigation Focus | Independent Scientific Status | Summary Finding |
| :--- | :---: | :--- |
| **1. Model Checkpoints** | **VERIFIED** | All 9 PyTorch checkpoints are authentic, structurally sound, non-trivially trained ($L_1$ weight shift from random initialization: 0.035–0.086), deterministic, and correctly mapped to $[B, 24, 13] \to [B, 1]$. |
| **2. Metric Reproduction** | **VERIFIED** | Metrics recalculated on the full held-out test partition ($N=19,455$ hours; 2024-07-29 to 2026-10-17). Exact model RMSE and $R^2$ confirmed. |
| **3. Data Leakage** | **VERIFIED (CLEAN)** | Absolute zero target leakage. Strict chronological disjointness between Train ($N=90,787$), Validation ($N=19,454$), and Test ($N=19,455$). Scaler fitted strictly on training data ($t < 90,787$). |
| **4. Baseline Comparisons** | **VERIFIED** | Persistence ($SWH_t$) and historical moving averages (MA-6h, MA-24h) independently evaluated. WaveLSTM beats persistence at +6h (+13.88% skill score). At +1h, persistence dominates due to physical wave inertia ($r_1 = 0.9995$). |
| **5. Generalization to Indian Coasts** | **PARTIALLY VERIFIED** | Validated for Mumbai Offshore ($18.50^\circ\text{N}, 72.50^\circ\text{E}$). The other 7 candidate nodes are zero-shot transfers without regional training or bathymetry; their multi-location accuracy is strictly **UNVERIFIED**. |
| **6. Real-Time Live Inference** | **VERIFIED** | Live Open-Meteo ingestion verified. Controlled perturbations confirm expected dynamic physical responses. Missing features, NaNs, and temporal gaps are strictly rejected. |
| **7. Scientific Plausibility** | **VERIFIED** | Extremely high $R^2$ ($0.9986$ at +1h) is mathematically and physically consistent with open-ocean gravity wave inertia ($r_1 = 0.99949$, $\text{Var}(y) = 0.51995\text{ m}^2$). |
| **8. Audit Discrepancies** | **FAILED CLAIM DETECTED** | **Correction**: The previous audit report claimed a +30.0% to +30.7% skill score at +24h based on a reported persistence RMSE of $0.2742\text{ m}$. Independent reproduction proves actual test-set persistence RMSE is **$0.1866\text{ m}$**, yielding a slightly negative RMSE skill (-1.90% to -6.06%), though models improve MAE by +4.71%. |

---

## 1. Checkpoint Verification

All 9 PyTorch checkpoints in `models/` were loaded on CPU with PyTorch 2.12.0. Each model was reconstructed using `src/models/` architectures and verified against the serialized weights and metadata.

### 1.1 Checkpoint Inspection Summary

| Horizon | Model Name | Checkpoint File | Metadata JSON | Trainable Parameters | Weight Mean | Weight Std | L1 Difference from Fresh Random Init | Repeatability (5 Runs) | Output Tensor Shape |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | `WaveANN` | `best_ann.pt` | `best_ann_metadata.json` | 50,881 | $+0.0032$ | $0.0793$ | $0.0463$ | Bitwise Identical | $[4, 1]$ |
| **+1h** | `WaveLSTM` | `best_lstm.pt` | `best_lstm_metadata.json` | 55,617 | $+0.0009$ | $0.0675$ | $0.0808$ | Bitwise Identical | $[4, 1]$ |
| **+1h** | `WaveCNN1D` | `best_cnn1d.pt` | `best_cnn1d_metadata.json` | 76,929 | $+0.0008$ | $0.0600$ | $0.0353$ | Bitwise Identical | $[4, 1]$ |
| **+6h** | `WaveANN` | `best_ann_h6.pt` | `best_ann_h6_metadata.json` | 50,881 | $+0.0033$ | $0.0798$ | $0.0468$ | Bitwise Identical | $[4, 1]$ |
| **+6h** | `WaveLSTM` | `best_lstm_h6.pt`| `best_lstm_h6_metadata.json` | 55,617 | $+0.0009$ | $0.0695$ | $0.0813$ | Bitwise Identical | $[4, 1]$ |
| **+6h** | `WaveCNN1D` | `best_cnn1d_h6.pt`| `best_cnn1d_h6_metadata.json`| 76,929 | $+0.0000$ | $0.0607$ | $0.0364$ | Bitwise Identical | $[4, 1]$ |
| **+24h**| `WaveANN` | `best_ann_h24.pt` | `best_ann_h24_metadata.json` | 50,881 | $-0.0002$ | $0.0879$ | $0.0540$ | Bitwise Identical | $[4, 1]$ |
| **+24h**| `WaveLSTM` | `best_lstm_h24.pt`| `best_lstm_h24_metadata.json`| 55,617 | $+0.0006$ | $0.0781$ | $0.0863$ | Bitwise Identical | $[4, 1]$ |
| **+24h**| `WaveCNN1D`| `best_cnn1d_h24.pt`| `best_cnn1d_h24_metadata.json`| 76,929 | $+0.0005$ | $0.0604$ | $0.0357$ | Bitwise Identical | $[4, 1]$ |

### 1.2 Forensic Weight Distribution Analysis
- **Non-Triviality Verification**: To confirm that the checkpoints were genuinely trained via gradient descent rather than initialized randomly, fresh model instances were initialized with identical random seeds and compared against the checkpoint weights. The mean absolute weight difference ($L_1$ norm) ranges from **0.0353 to 0.0863**, showing significant parameter migration along the loss surface.
- **Numerical Health**: All tensors were audited for extreme values: zero `NaN` values and zero `Inf` values were detected. Weight distributions are centered near zero with standard deviations between 0.060 and 0.088, typical of Xavier/Kaiming converged networks with Batch Normalization.
- **Deterministic Forward Pass**: Passing an identical synthetic tensor $\mathbf{X} \in \mathbb{R}^{4 \times 24 \times 13}$ through all 9 models 5 consecutive times produced bitwise identical outputs across all runs.

---

## 2. Independently Reproduced Model Metrics

Metrics were recomputed from scratch using the held-out test partition of `data/processed/era5_wave_offshore_timeseries.parquet` ($N = 19,455$ contiguous hourly time steps from **2024-07-29 09:00 UTC to 2026-10-17 23:00 UTC**).

Ground truth target properties on the held-out test partition:
- $\text{Mean}(y) = 1.2763\text{ m}$
- $\text{Std}(y) = 0.7211\text{ m}$
- $\text{Var}(y) = 0.51995\text{ m}^2$
- $\text{Min}(y) = 0.3753\text{ m}$, $\text{Max}(y) = 4.4000\text{ m}$

### 2.1 Complete Independent Metric Evaluation Table

| Horizon | Model / Baseline | MAE (m) | MSE ($\text{m}^2$) | RMSE (m) | $R^2$ Score | MAPE (%) | Explained Variance | RMSE Skill vs Persistence (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | **Naive Persistence** | **0.0131** | **0.00045** | **0.0213** | **0.9991** | **1.20%** | **0.9991** | **Baseline (0.00%)** |
| **+1h** | Historical MA (6h) | 0.0379 | 0.00303 | 0.0550 | 0.9942 | 3.50% | 0.9942 | -158.39% |
| **+1h** | Historical MA (24h) | 0.0774 | 0.01230 | 0.1109 | 0.9764 | 6.79% | 0.9764 | -421.15% |
| **+1h** | `WaveANN` | 0.0349 | 0.00217 | 0.0465 | 0.9958 | 3.38% | 0.9958 | -118.81% |
| **+1h** | **`WaveLSTM`** | **0.0182** | **0.00073** | **0.0271** | **0.9986** | **1.64%** | **0.9986** | **-27.20%** |
| **+1h** | `WaveCNN1D` | 0.0426 | 0.00345 | 0.0587 | 0.9934 | 3.86% | 0.9934 | -176.08% |
| **+6h** | Naive Persistence | 0.0624 | 0.00798 | 0.0893 | 0.9846 | 5.77% | 0.9846 | Baseline (0.00%) |
| **+6h** | Historical MA (6h) | 0.0749 | 0.01126 | 0.1061 | 0.9783 | 6.90% | 0.9783 | -18.80% |
| **+6h** | Historical MA (24h) | 0.0992 | 0.02025 | 0.1423 | 0.9610 | 8.61% | 0.9610 | -59.30% |
| **+6h** | `WaveANN` | 0.0574 | 0.00664 | 0.0815 | 0.9872 | 5.05% | 0.9872 | +8.77% |
| **+6h** | **`WaveLSTM`** | **0.0533** | **0.00592** | **0.0769** | **0.9886** | **4.71%** | **0.9886** | **+13.88%** |
| **+6h** | `WaveCNN1D` | 0.0679 | 0.00887 | 0.0942 | 0.9829 | 6.23% | 0.9830 | -5.43% |
| **+24h**| Naive Persistence | 0.1286 | 0.03482 | 0.1866 | 0.9330 | 10.98% | 0.9331 | Baseline (0.00%) |
| **+24h**| Historical MA (6h) | 0.1376 | 0.03936 | 0.1984 | 0.9243 | 11.76% | 0.9243 | -6.32% |
| **+24h**| Historical MA (24h) | 0.1638 | 0.05480 | 0.2341 | 0.8946 | 13.94% | 0.8946 | -25.45% |
| **+24h**| **`WaveANN`** | **0.1225** | 0.03686 | 0.1920 | 0.9291 | 10.59% | 0.9291 | -2.89% (+4.71% MAE skill) |
| **+24h**| **`WaveLSTM`** | 0.1229 | 0.03916 | 0.1979 | 0.9247 | **10.33%** | 0.9247 | -6.06% (+4.45% MAE skill) |
| **+24h**| **`WaveCNN1D`** | 0.1362 | **0.03615** | **0.1901** | **0.9305** | 12.55% | **0.9319** | **-1.90%** (-5.88% MAE skill) |

---

## 3. Data Leakage Forensic Investigation

A rigorous temporal and feature audit was performed to evaluate whether test-set accuracy was artificially inflated by data contamination.

```
CHRONOLOGICAL DATA SPLIT PARTITIONING (Total N = 129,696 Hours):

[0                                                 90,786] [90,787           110,240] [110,241           129,695]
├─────────────────────────────────────────────────────────┼──────────────────────────┼──────────────────────────┤
│             TRAINING PARTITION (70%)                    │  VALIDATION SET (15%)    │     TESTING SET (15%)    │
│            2012-01-01 to 2022-05-10                     │ 2022-05-10 to 2024-07-29 │ 2024-07-29 to 2026-10-17 │
│            Target Range: [24 .. 90,786]                 │ Targets: [90,787..110,240]│ Targets:[110,241..129,695]│
└─────────────────────────────────────────────────────────┴──────────────────────────┴──────────────────────────┘
                              ▲                                          ▲
                              │                                          │
                  TARGET BOUNDARY IS STRICT                  TARGET BOUNDARY IS STRICT
                 MAX(Train Target) = 90,786                 MAX(Val Target) = 110,240
                 MIN(Val Target)   = 90,787                 MIN(Test Target) = 110,241
                 Zero Target Overlap                        Zero Target Overlap
```

### 3.1 Forensic Leakage Check Results

1. **Target Disjointness Across Splits**:
   - $\text{Train Targets} \cap \text{Val Targets} = \emptyset$ (0 intersecting samples).
   - $\text{Val Targets} \cap \text{Test Targets} = \emptyset$ (0 intersecting samples).
   - $\text{Train Targets} \cap \text{Test Targets} = \emptyset$ (0 intersecting samples).
   - **Verdict**: Target data leakage is **strictly non-existent**.

2. **Feature Scaler Fitting**:
   - `scaler.pkl` was fitted using `TimeSeriesScaler.fit(df.iloc[:90787])`.
   - Feature means $\mu$ and standard deviations $\sigma$ were derived strictly from the 70% training partition.
   - Test data ($t \ge 110,241$) was normalized strictly using the frozen training parameters ($\mu = 1.4870\text{ m}, \sigma = 0.9854\text{ m}$).
   - **Verdict**: Scaler fitting exhibits **zero future information leakage**.

3. **Sliding-Window Historical Overlap vs Operational Reality**:
   - For early test samples (e.g. index $0$ of the test set, corresponding to anchor $t = 110,241 - H$), the 24-hour historical input sequence $\mathbf{X} \in \mathbb{R}^{24 \times 13}$ includes observations from $[t - 23, \dots, t]$.
   - Some of these historical observations fall into the calendar window of the validation split.
   - **Legitimate Context vs Leakage**: In operational time-series forecasting, providing historical observations up to anchor time $t$ to forecast future time $t + H$ is **legitimate historical context**. The model is predicting a completely unseen future target using only known past states. Leakage occurs only if future targets are fed into past inputs, which this architecture strictly prohibits ($\text{Input Window End} \le \text{Anchor Time} < \text{Target Time}$).

4. **Circular & Cyclical Feature Engineering**:
   - Sine/cosine transformations ($\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{month}/12)$, $\sin(2\pi \cdot \text{direction}/360)$) are instantaneous point-in-time mathematical transformations.
   - No centered moving averages, lookahead smoothing filters, or future interpolation windows were utilized in preprocessing.
   - **Verdict**: Feature engineering is **100% causal and leak-free**.

---

## 4. Baseline Comparisons & The +1h Paradox

### 4.1 Persistence Autocorrelation Dynamics
The empirical autocorrelation function (ACF) of significant wave height ($SWH$) reveals the physical mechanism governing model performance:

$$\rho(1\text{h}) = 0.99949 \quad\Big|\quad \rho(6\text{h}) = 0.99108 \quad\Big|\quad \rho(24\text{h}) = 0.96086$$

```
TEMPORAL AUTOCORRELATION & PHYSICAL MASS INERTIA:

Lag 1 Hour  : [████████████████████████████████████████]  r = 0.99949 (Virtually Static)
Lag 6 Hours : [██████████████████████████████████████░]  r = 0.99108 (Slight Swell Evolution)
Lag 24 Hours: [██████████████████████████████████░░░░]  r = 0.96086 (Synoptic Meteorological Shift)
```

### 4.2 Why Deep Learning Performs Worse at +1 Hour
At the 1-hour horizon, the persistence forecast $\hat{y}_{t+1} = y_t$ achieves an RMSE of **$0.0213\text{ m}$** ($R^2 = 0.9991$). 

Deep learning models achieve:
- `WaveLSTM`: $\text{RMSE} = 0.0271\text{ m}$ ($R^2 = 0.9986$), Skill Score = **$-27.20\%$**
- `WaveANN`: $\text{RMSE} = 0.0465\text{ m}$ ($R^2 = 0.9958$), Skill Score = **$-118.81\%$**
- `WaveCNN1D`: $\text{RMSE} = 0.0587\text{ m}$ ($R^2 = 0.9934$), Skill Score = **$-176.08\%$**

**Scientific Explanation**:
1. **Physical Ocean Inertia**: Ocean surface gravity waves represent massive mechanical energy distributions across vast surface areas. Wave height cannot physically change by more than a few centimeters in 60 minutes under typical offshore conditions.
2. **Neural Optimization Noise**: Any deep learning model must approximate the identity function ($f(\mathbf{X}) \approx x_{24, 0}$) through hundreds of matrix multiplications, nonlinear activations, dropout perturbations ($p=0.2$), and floating-point inverse transformations.
3. **Variance Accumulation**: This approximation adds an empirical error variance of $\sim 0.0058\text{ m}$ ($\approx 5.8\text{ mm}$) on top of the physical state. Consequently, naive persistence is mathematically and physically difficult to beat at +1h.

### 4.3 Why Deep Learning Dominates at +6 Hours
At +6h, physical autocorrelation decays ($\rho = 0.99108$). Wind-forcing and wave dispersion begin altering the sea state:
- Persistence RMSE degrades by $+320\%$ to **$0.0893\text{ m}$**.
- **`WaveLSTM` captures historical wind momentum**, achieving an RMSE of **$0.0769\text{ m}$** ($R^2 = 0.9886$).
- `WaveLSTM` achieves a **$+13.88\%$ positive skill score over persistence**, proving genuine deep learning utility.

---

## 5. Model Generalization & Multi-Location Validity

The project provides predictions across 8 Indian coastal monitoring nodes:
1. **Mumbai Offshore** ($18.50^\circ\text{N}, 72.50^\circ\text{E}$) — *Primary Reference Node*
2. **Mormugao / Goa** ($15.40^\circ\text{N}, 73.70^\circ\text{E}$)
3. **Mangalore Port** ($12.90^\circ\text{N}, 74.75^\circ\text{E}$)
4. **Kochi Coast** ($9.95^\circ\text{N}, 76.15^\circ\text{E}$)
5. **Kanyakumari** ($8.05^\circ\text{N}, 77.55^\circ\text{E}$)
6. **Chennai Port** ($13.10^\circ\text{N}, 80.35^\circ\text{E}$)
7. **Visakhapatnam Port** ($17.65^\circ\text{N}, 83.35^\circ\text{E}$)
8. **Paradip Port** ($20.25^\circ\text{N}, 86.70^\circ\text{E}$)

### 5.1 Geographic Domain Shift & Climatological Hazards
The 9 neural network models and `scaler.pkl` were trained **exclusively** on single-point reanalysis at Mumbai Offshore. Applying these models to the remaining 7 locations represents a **zero-shot spatial transfer**:

1. **Arabian Sea vs Bay of Bengal Asymmetry**:
   - Mumbai, Goa, Mangalore, and Kochi are on the western coast (Arabian Sea), where wave dynamics are governed by the Southwest Monsoon (June–September) with energetic westerly swells.
   - Chennai, Visakhapatnam, and Paradip are on the eastern coast (Bay of Bengal), governed by the Northeast Monsoon (October–December) and frequent severe cyclonic storms with intense local surges.
2. **Tri-Junction Southern Exposure (Kanyakumari)**:
   - Kanyakumari is exposed directly to long-period Southern Ocean swell traveling thousands of kilometers across the Southern Hemisphere, creating complex cross-sea states.
3. **Nearshore Bathymetric Shoaling**:
   - Mumbai Offshore is located ~80km offshore in deep continental shelf waters.
   - Candidate nodes located 6km to 15km from shore experience shallow-water shoaling, bottom friction, and wave refraction ($\lambda = \frac{g T^2}{2\pi} \tanh(2\pi d / \lambda)$). The deep learning models receive no bathymetric depth feature ($d$), making local shoaling physics invisible to the neural network.

### 5.2 Independent Finding on Regional Validity
- **Finding**: No regional training datasets or held-out validation records exist in the repository for the 7 non-Mumbai stations.
- **Code Audit**: The backend codebase in `src/intelligence/nodes.py` explicitly designates these 7 stations as `EXPERIMENTAL_UNVALIDATED`.
- **Scientific Audit Verdict**: Multi-location scientific accuracy is **UNVERIFIED**. The models cannot legitimately claim calibrated accuracy outside Mumbai Offshore without local in-situ buoy calibrations (e.g. INCOIS OOS waverider buoy deployments).

---

## 6. Live Real-Time Inference Verification

The real-time operational forecasting pipeline was tested with live network calls to the configured Open-Meteo REST service (`https://marine-api.open-meteo.com` and `https://api.open-meteo.com`).

```
LIVE INFERENCE EXECUTION PIPELINE AUDIT:

[ Open-Meteo REST API ]
  │ Latency: 1.34s | Status: 200 OK | Rows: 96 | Fresh Telemetry
  ▼
[ validate_and_prepare_input_window() ]
  │ 24 Hourly Rows Isolated | Anchor: 2026-11-16 11:00 UTC | Zero NaNs | No Gaps
  ▼
[ TimeSeriesScaler.transform() ]
  │ Mean: 1.4870 | Scale: 0.9854 | Training Distribution Applied
  ▼
[ PyTorch WaveLSTM Forward Pass ]
  │ Checkpoint: best_lstm_h6.pt | In: [1, 24, 13] | Out Scaled: -0.6481
  ▼
[ TimeSeriesScaler.inverse_transform_target() ]
  │ Inverse Scaled Output: 0.7630 m
  ▼
[ Controlled Perturbation Check ]
  │ Swell Pulse (+1.5m) -> 2.3776 m (+1.6146 m)
  │ Wind Surge (+15 m/s) -> 1.0389 m (+0.2760 m)
  │ Calm Decay (0.20m)   -> 0.4708 m (-0.2921 m)
```

### 6.1 Controlled Sensitivity Tests

| Test Condition | Input Modification | Output Prediction (m) | Delta from Baseline (m) | Physical Plausibility |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Live Input** | Live real-time observations | $0.7630\text{ m}$ | Baseline ($0.000\text{ m}$) | Verified |
| **Swell Pulse** | $+1.5\text{ m}$ wave height over past 6h | $2.3776\text{ m}$ | $+1.6146\text{ m}$ | **PASS** (Strong positive swell response) |
| **Wind Acceleration** | $+15\text{ m/s}$ wind over past 12h | $1.0389\text{ m}$ | $+0.2760\text{ m}$ | **PASS** (Physical wind-wave growth) |
| **Calm Sea Decay** | Wave height clamped to $0.20\text{ m}$ | $0.4708\text{ m}$ | $-0.2921\text{ m}$ | **PASS** (Monotonic wave decay) |

### 6.2 Error Handling & Negative Testing
- **Missing Required Feature**: Dropping column `swh` triggered `KeyError: "Required physical column 'swh' is missing from observations."` — **PASS**.
- **Missing NaN Value**: Injecting `np.nan` into the active 24-hour observation window triggered `ValueError: "Missing (NaN) values detected in column 'swh'. Forecast aborted."` — **PASS**.
- **Irregular Timestamp Gap**: Dropping a row inside the active 24-hour window triggered `ValueError: "Irregular or missing hourly timestamps detected in input window... Silent imputation is disabled for scientific integrity."` — **PASS**.
- **Determinism Check**: 5 consecutive live inference runs on the verified window produced identical output to 16 decimal places (`0.7629515528678894`).

---

## 7. Scientific Validity of Performance Claims

### 7.1 Mathematical Consistency of $R^2$ and RMSE
The coefficient of determination $R^2$ is defined as:

$$R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)} = 1 - \frac{\text{RMSE}^2}{\text{Var}(y)}$$

Using our independently calculated ground truth variance on the test set ($\text{Var}(y) = 0.51995\text{ m}^2$):
- **WaveLSTM at +1h**: $\text{RMSE} = 0.02706\text{ m} \implies \text{MSE} = 0.0007323\text{ m}^2$
  $$R^2 = 1 - \frac{0.0007323}{0.51995} = 1 - 0.001408 = \mathbf{0.99859}$$
- **WaveLSTM at +6h**: $\text{RMSE} = 0.07694\text{ m} \implies \text{MSE} = 0.005920\text{ m}^2$
  $$R^2 = 1 - \frac{0.005920}{0.51995} = 1 - 0.011386 = \mathbf{0.98861}$$
- **WaveCNN1D at +24h**: $\text{RMSE} = 0.19013\text{ m} \implies \text{MSE} = 0.03615\text{ m}^2$
  $$R^2 = 1 - \frac{0.03615}{0.51995} = 1 - 0.069525 = \mathbf{0.93048}$$

**Conclusion**: All reported $R^2$ and RMSE pairs are **100% mathematically consistent**.

---

## 8. Discrepancies with Previous Technical Audit

This independent investigation revealed several discrepancies with the claims presented in `CP_COMPLETE_TECHNICAL_AUDIT.md`:

### 8.1 Discrepancy Table

| Item | Previous Audit Claim | Independent Empirical Finding | Status | Auditor Analysis |
| :--- | :--- | :--- | :---: | :--- |
| **+24h Persistence Baseline RMSE** | $\text{RMSE} = 0.2742\text{ m}$, $\text{MAE} = 0.1845\text{ m}$, $R^2 = 0.8550$ | **$\text{RMSE} = 0.1866\text{ m}$**, **$\text{MAE} = 0.1286\text{ m}$**, **$R^2 = 0.9330$** | **FAILED (False Claim in Previous Audit)** | The previous audit reported an unverified persistence RMSE of 0.2742m. On the actual held-out test partition ($N=19,455$), persistence RMSE is 0.1866m. |
| **+24h Deep Learning Skill Score** | Claimed **$+30.0\%$ to $+30.7\%$** skill score over persistence | Actual RMSE skill score is **$-1.90\%$ to $-6.06\%$** (WaveANN/LSTM achieve $+4.7\%$ MAE skill) | **FAILED (False Claim in Previous Audit)** | Because actual persistence RMSE is 0.1866m rather than 0.2742m, the +30% RMSE skill score claim is refuted. Models improve MAE but slightly lag on squared error. |
| **Dataset Length & Span** | Claimed 394,464 hourly records (1979–2023, 44 years) | Actual parquet file contains **129,696 hourly records (2012–2026, 14.8 years)** | **CORRECTED** | The active processed dataset spans 2012 to 2026. The 44-year figure was an erroneous documentation artifact. |
| **Historical Moving Average Baselines** | Omitted | Independently implemented MA-6h and MA-24h | **IMPROVED** | Moving averages confirm that smoothing degrades forecast accuracy compared to persistence and deep learning. |

---

## 9. Recommended Improvements

1. **Residual Difference Formulation ($\Delta SWH$)**:
   - Rather than training neural networks to predict absolute wave height $SWH_{t+H}$, train them to predict the temporal delta:
     $$\Delta SWH = SWH_{t+H} - SWH_t$$
   - At inference time: $\hat{SWH}_{t+H} = SWH_t + \hat{\Delta SWH}$. This mathematically guarantees that in calm regimes where $\hat{\Delta} \to 0$, model performance defaults gracefully to persistence without penalty.
2. **Regional In-Situ Buoy Calibrations**:
   - Ingest physical moored buoy time series from INCOIS (such as Western Arabian Sea and Eastern Bay of Bengal buoys) to train localized transfer adapters or fine-tune checkpoint heads for the 7 candidate nodes.
3. **Bathymetry & Distance-to-Coast Features**:
   - Append coastal water depth ($d$) from GEBCO or ETOPO digital elevation models to enable the models to capture shallow-water wave dissipation.
4. **Probabilistic Uncertainty Estimation**:
   - Integrate Monte Carlo Dropout or Deep Ensembles to output 10th and 90th percentile prediction bands rather than purely deterministic scalar predictions.

---

## 10. Final Confidence Assessment & Classification

### 10.1 Systematic Claim Classification

| # | Scientific Audit Claim | Verification Status | Final Audit Conclusion |
| :---: | :--- | :---: | :--- |
| **1** | All 9 PyTorch checkpoints contain genuine, non-zero, trained weights. | **VERIFIED** | Checkpoint parameters show distinct convergence patterns ($L_1$ weight shift 0.035–0.086) and zero NaNs/Infs. |
| **2** | Deep learning inference is dynamic and responds to physical sensor variations. | **VERIFIED** | Perturbation tests confirm positive shifts for swell/wind acceleration and negative shifts for calm decay. |
| **3** | The data pipeline is free from target and feature leakage. | **VERIFIED** | Strict chronological separation: Train ($t \le 90,786$), Val ($90,787 \le t \le 110,240$), Test ($t \ge 110,241$). Scaler fitted strictly on training data. |
| **4** | Deep learning outperforms persistence at the +6-hour horizon. | **VERIFIED** | WaveLSTM achieves **$\text{RMSE} = 0.0769\text{ m}$** vs Persistence **$\text{RMSE} = 0.0893\text{ m}$** (**$+13.88\%$ Skill Score**). |
| **5** | Deep learning outperforms persistence by +30% at the +24-hour horizon. | **FAILED** | **Refuted**: On the true held-out test partition, persistence RMSE is $0.1866\text{ m}$. Deep learning models achieve RMSE $0.1901\text{–}0.1979\text{ m}$ (RMSE skill $-1.9\%$ to $-6.1\%$; MAE skill $+4.7\%$). |
| **6** | Deep learning underperforms persistence at the +1-hour horizon. | **VERIFIED** | High wave inertia ($\rho_1 = 0.9995$) makes persistence mathematically tough to beat. Deep learning suffers from $\sim 5.8\text{ mm}$ optimization noise. |
| **7** | Real-time live inference connects to live external numerical feeds. | **VERIFIED** | Open-Meteo REST API ingests live ECMWF WAM and IFS data for Mumbai Offshore; fallback to offline demo occurs only when network fails. |
| **8** | Models provide scientifically verified multi-location accuracy across India. | **NOT VERIFIED** | Validated exclusively at Mumbai Offshore. The 7 other coastal locations are zero-shot transfers without local validation; correctly labeled `EXPERIMENTAL_UNVALIDATED`. |

### 10.2 Auditor Certification
The OceanCast AI / Coastal Prediction codebase is a **genuine, operational, and non-simulated deep learning system**. It features authentic PyTorch neural network checkpoints, an operational real-time atmospheric/marine data ingestion pipeline, and zero data leakage. With the correction of the +24h persistence benchmark and the clear scientific demarcation of regional candidate nodes, the platform meets rigorous standards of machine learning research integrity.
