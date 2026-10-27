"""
Visualization Modules for Wave Forecasting:
- Training and validation loss convergence curves
- Actual vs. predicted time-series comparisons
- Residual error distributions and scatter diagnostics
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


# Clean styling palette
MODEL_COLORS = {
    "ANN": "#d35400",       # Rust Orange
    "LSTM": "#2980b9",      # Marine Blue
    "1D-CNN": "#27ae60",    # Emerald Green
    "WaveANN": "#d35400",
    "WaveLSTM": "#2980b9",
    "WaveCNN1D": "#27ae60",
}


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot training and validation loss curves across models side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # 1. Training Loss Subplot
    for model_name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = MODEL_COLORS.get(model_name, None)
        axes[0].plot(epochs, hist["train_loss"], label=model_name, color=color, linewidth=1.8)

    axes[0].set_title("Training Loss Convergence (MSE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss (Normalized MSE)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True, fontsize=10)

    # 2. Validation Loss Subplot
    for model_name, hist in histories.items():
        epochs = range(1, len(hist["val_loss"]) + 1)
        color = MODEL_COLORS.get(model_name, None)
        axes[1].plot(epochs, hist["val_loss"], label=model_name, color=color, linewidth=1.8)
        # Highlight best validation epoch
        best_idx = int(np.argmin(hist["val_loss"]))
        best_val = hist["val_loss"][best_idx]
        axes[1].scatter([best_idx + 1], [best_val], color=color, s=50, zorder=5)

    axes[1].set_title("Validation Loss Convergence (MSE)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, fontsize=10)

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Ocean Wave Height (SWH) Forecast Comparison on Untouched Test Set",
    sample_window: int = 350,
):
    """
    Plot actual observed significant wave height vs model predictions in physical units (meters).
    """
    y_true = np.asarray(y_true).flatten()
    pts = min(len(y_true), sample_window)

    if timestamps is not None and len(timestamps) >= pts:
        x_axis = pd.to_datetime(timestamps[:pts])
        x_label = "Date & Time (UTC)"
    else:
        x_axis = np.arange(pts)
        x_label = "Forecast Time Steps (Hours)"

    plt.figure(figsize=(14, 5.5))

    # Observed ERA5 ground truth
    plt.plot(
        x_axis,
        y_true[:pts],
        label="ERA5 Ground Truth SWH",
        color="#2c3e50",
        linewidth=2.2,
        alpha=0.9,
    )

    # Model predictions
    for name, pred in predictions.items():
        p_arr = np.asarray(pred).flatten()[:pts]
        color = MODEL_COLORS.get(name, None)
        plt.plot(
            x_axis,
            p_arr,
            label=f"{name} Forecast",
            linestyle="--",
            linewidth=1.6,
            color=color,
            alpha=0.9,
        )

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel("Significant Wave Height (m)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_residuals(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Diagnostic residual analysis:
    - Left: Kernel Density Estimation (KDE) error residual distribution (y_pred - y_true)
    - Right: Residual errors vs Observed Wave Height scatter (evaluating calm vs extreme errors)
    """
    y_true = np.asarray(y_true).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: KDE distribution of errors
    for name, pred in predictions.items():
        err = np.asarray(pred).flatten() - y_true
        color = MODEL_COLORS.get(name, None)
        sns.kdeplot(err, label=f"{name} (Bias: {np.mean(err):+.3f}m)", color=color, fill=True, alpha=0.15, ax=axes[0])

    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Forecast Error Residuals Density (Pred - Actual)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Residual Error (meters)", fontsize=11)
    axes[0].set_ylabel("Probability Density", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True, fontsize=9)

    # Right: Residuals vs Actual Wave Height
    for name, pred in predictions.items():
        err = np.asarray(pred).flatten() - y_true
        color = MODEL_COLORS.get(name, None)
        # Sample 1000 points for clear visualization
        sub_idx = np.random.choice(len(y_true), size=min(1000, len(y_true)), replace=False)
        axes[1].scatter(y_true[sub_idx], err[sub_idx], label=name, color=color, alpha=0.35, s=15)

    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_title("Residuals vs. Actual Wave Height (m)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Actual Wave Height (meters)", fontsize=11)
    axes[1].set_ylabel("Residual Error (meters)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, fontsize=9)

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    metric_name: str = "rmse",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Bar plot comparing a specific metric across models.
    """
    plt.figure(figsize=(7, 4))
    sns.barplot(data=metrics_df, x="model", y=metric_name, palette="Blues_d")
    plt.title(f"Model Comparison: {metric_name.upper()}", fontsize=12, fontweight="bold")
    plt.ylabel(metric_name.upper())
    plt.xlabel("Model")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_live_forecast_trajectory(
    prepared_df: pd.DataFrame,
    anchor_time: pd.Timestamp,
    target_time: pd.Timestamp,
    forecast_swh: float,
    persistence_swh: float,
    model_name: str,
    horizon_hours: int,
    external_nwp_swh: Optional[float] = None,
    dark_mode: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Publication-grade dual-panel time-series visualization for live operational forecasts:
    - Upper panel: 24h observed SWH sequence, anchor state t, DL model forecast t+H,
      external NWP forecast t+H, and shaded WMO sea state categories.
    - Lower panel: 10m wind velocity and mean wave period (Tp) over the same 24 hours.
    """
    times = pd.to_datetime(prepared_df["valid_time"])
    swh_vals = prepared_df["swh"].values

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # ----------------- Upper Panel: Wave Dynamics -----------------
    # Shaded WMO Sea State Bands
    ax1.axhspan(0.0, 0.5, color="#0284c7", alpha=0.07, label="_nolegend_")
    ax1.axhspan(0.5, 1.25, color="#10b981", alpha=0.07, label="_nolegend_")
    ax1.axhspan(1.25, 2.5, color="#eab308", alpha=0.07, label="_nolegend_")
    ax1.axhspan(2.5, 4.0, color="#f97316", alpha=0.07, label="_nolegend_")

    # Observed 24h SWH curve
    ax1.plot(
        times,
        swh_vals,
        color="#38bdf8",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Observed Significant Wave Height ($H_s$)",
        zorder=3,
    )

    # Observation anchor marker
    ax1.scatter(
        [anchor_time],
        [persistence_swh],
        color="#f59e0b",
        s=110,
        zorder=6,
        edgecolor="#ffffff",
        linewidth=1.2,
        label=f"Observation Anchor $t$ ({anchor_time.strftime('%H:00 UTC')} : {persistence_swh:.3f} m)",
    )

    # Deep learning forecast marker
    ax1.scatter(
        [target_time],
        [forecast_swh],
        color="#10b981",
        s=170,
        marker="*",
        zorder=7,
        edgecolor="#ffffff",
        linewidth=1.0,
        label=f"{model_name} Forecast $t+{horizon_hours}\\text{{h}}$ ({forecast_swh:.3f} m)",
    )

    # Trajectory projection line
    ax1.plot(
        [anchor_time, target_time],
        [persistence_swh, forecast_swh],
        color="#10b981",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=4,
    )

    # External NWP forecast comparison point if available
    if external_nwp_swh is not None:
        ax1.scatter(
            [target_time],
            [external_nwp_swh],
            color="#c084fc",
            s=120,
            marker="D",
            zorder=7,
            edgecolor="#ffffff",
            linewidth=1.0,
            label=f"External NWP (ECMWF WAM) $t+{horizon_hours}\\text{{h}}$ ({external_nwp_swh:.3f} m)",
        )
        ax1.plot(
            [anchor_time, target_time],
            [persistence_swh, external_nwp_swh],
            color="#c084fc",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=4,
        )

    # Persistence baseline reference line
    ax1.axhline(
        persistence_swh,
        color="#94a3b8",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=f"Persistence Baseline: {persistence_swh:.3f} m",
    )

    max_swh = max(float(swh_vals.max()), forecast_swh, external_nwp_swh or 0.0)
    ax1.set_ylim(0.0, max(2.5, max_swh * 1.35))
    ax1.set_ylabel("Wave Height $H_s$ (m)", fontsize=11, color="#e2e8f0" if dark_mode else "#0f172a")
    ax1.set_title(
        f"Offshore Wave Forecasting: 24h Ingestion Context vs {horizon_hours}h-Ahead Prognosis",
        fontsize=12,
        fontweight="bold",
        color="#e2e8f0" if dark_mode else "#0f172a",
    )

    # ----------------- Lower Panel: Atmospheric & Period Drivers -----------------
    if "wind_speed" in prepared_df.columns:
        ax2.plot(
            times,
            prepared_df["wind_speed"],
            color="#f59e0b",
            linewidth=1.8,
            label="10m Wind Speed (m/s)",
        )
    ax2.set_ylabel("Wind Speed (m/s)", fontsize=10, color="#f59e0b")

    if "mwp" in prepared_df.columns:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            times,
            prepared_df["mwp"],
            color="#34d399",
            linewidth=1.6,
            linestyle="-.",
            label="Mean Wave Period $T_p$ (s)",
        )
        ax2_twin.set_ylabel("Period $T_p$ (s)", fontsize=10, color="#34d399")
        ax2_twin.tick_params(colors="#34d399")
        for spine in ax2_twin.spines.values():
            spine.set_color("#334155" if dark_mode else "#cbd5e1")

    ax2.set_xlabel("Time (UTC)", fontsize=11, color="#e2e8f0" if dark_mode else "#0f172a")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))

    # Theme and styling
    if dark_mode:
        fig.patch.set_facecolor("#0b111e")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
            for spine in ax.spines.values():
                spine.set_color("#334155")
        ax1.legend(
            loc="upper left",
            facecolor="#1e293b",
            edgecolor="#334155",
            labelcolor="#e2e8f0",
            fontsize=9.5,
        )
        ax2.legend(
            loc="upper left",
            facecolor="#1e293b",
            edgecolor="#334155",
            labelcolor="#e2e8f0",
            fontsize=9.0,
        )
    else:
        for ax in [ax1, ax2]:
            ax.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=9.5)
        ax2.legend(loc="upper left", fontsize=9.0)

    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig

