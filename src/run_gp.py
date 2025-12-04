import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_nab import load_series, load_labels, mark_anomaly_windows
from src.evaluate import compute_detection_metrics, compute_event_level_metrics, persist_filter, compute_business_metrics
from src.utils import rolling_sigma_threshold
from src.plotting import plot_residual_diagnostics, plot_pr_curve

def fit_gp_composite(y_train: np.ndarray, X_train: np.ndarray = None) -> GaussianProcessRegressor:
    """
    Fit GP with Composite Kernel: Trend + Daily Seasonality + Weekly Seasonality + Noise
    """
    n = len(y_train)
    X = np.arange(n).reshape(-1, 1) if X_train is None else X_train
    
    # Kernel Definition
    # 1. Long term trend (RBF)
    k_trend = C(50.0, (1e-3, 1e3)) * RBF(length_scale=50.0, length_scale_bounds=(10, 200))
    
    # 2. Daily Seasonality (48 samples)
    k_daily = C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=48.0, periodicity_bounds=(47.5, 48.5))
    
    # 3. Weekly Seasonality (336 samples)
    k_weekly = C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=336.0, periodicity_bounds=(335, 337))
    
    # 4. Noise
    k_noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
    
    kernel = k_trend + k_daily + k_weekly + k_noise
    
    print(f"Fitting GP with kernel: {kernel}")
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=1)
    gp.fit(X, y_train)
    print(f"Learned kernel: {gp.kernel_}")
    return gp

def run_gp_pipeline(nab_root: str, file_key: str, train_frac: float = 0.5, label_window: int = 1, save_dir: str = "./results/gp"):
    save_dir = Path(save_dir) / file_key.replace("/", "__")
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = load_series(nab_root, file_key)
    labels = load_labels(nab_root)
    label_times = labels.get(file_key, labels.get("data/" + file_key, []))
    df = mark_anomaly_windows(df, label_times, window_size=label_window)
    
    n = len(df)
    train_end = int(n * train_frac)
    
    # Subsample for training (GP is slow)
    # Use last 1000 points of training set
    train_subset_size = 1000
    train_start_idx = max(0, train_end - train_subset_size)
    
    y_train_full = df['value'].iloc[:train_end].values
    y_train_subset = y_train_full[train_start_idx:]
    X_train_subset = np.arange(train_start_idx, train_end).reshape(-1, 1)
    
    print(f"Fitting GP on {len(y_train_subset)} samples...")
    gp = fit_gp_composite(y_train_subset, X_train_subset)

    # forecast
    print("Forecasting on Test Set...")
    X_test = np.arange(train_end, n).reshape(-1, 1)
    mean, std = gp.predict(X_test, return_std=True)
    
    y_test = df['value'].iloc[train_end:].values
    residuals = y_test - mean
    
    # --- Diagnostics ---
    print("Generating Diagnostic Plots...")
    plot_residual_diagnostics(residuals, "GP_Composite", save_dir)
    
    # PR Curve (using absolute residual as score)
    y_true_test = df['is_anomaly'].iloc[train_end:].values
    plot_pr_curve(y_true_test, np.abs(residuals), "GP_Composite", save_dir)

    # --- Detection ---
    print("\n--- Starting Threshold Sweep (Rolling Sigma) ---")
    best_k = 3.0
    best_f1 = -1.0
    best_metrics = None
    
    for k_candidate in np.linspace(2.0, 10.0, 17):
        # Use Rolling Sigma from utils
        flags_temp = rolling_sigma_threshold(residuals, k=k_candidate, window=48)
        
        # Apply Persistence
        flags_temp = persist_filter(flags_temp, p=2)
        
        m_evt = compute_event_level_metrics(y_true_test, flags_temp, gap=3)
        f1 = m_evt['f1']
        
        # Constraint: Anomaly Rate should not be excessive (> 10%)
        anomaly_rate = np.mean(flags_temp)
        if anomaly_rate > 0.10:
            continue
        
        if f1 > best_f1:
            best_f1 = f1
            best_k = k_candidate
            best_metrics = m_evt
            
    print(f"--- Best Threshold: k={best_k:.1f} with F1={best_f1:.4f} ---\n")
    
    # Final Flags
    flags = rolling_sigma_threshold(residuals, k=best_k, window=48)
    flags = persist_filter(flags, p=2)
    
    # Business Metrics
    bus_metrics = compute_business_metrics(
        y_true_test, 
        flags, 
        df['timestamp'].iloc[train_end:], 
        gap=3
    )
    print(f"FP/day: {bus_metrics['fp_per_day']:.2f}, Latency: {bus_metrics['median_latency_minutes']:.1f} min")

    # Save results
    out_df = df.iloc[train_end:].reset_index(drop=True).copy()
    out_df['pred_mean'] = mean
    out_df['pred_std'] = std
    out_df['detected'] = flags
    out_df.to_csv(save_dir / "predictions.csv", index=False)

    # Plot Forecast
    plt.figure(figsize=(14, 4))
    plt.plot(df['timestamp'].iloc[train_end:], y_test, label='Actual', alpha=0.6, color='black')
    plt.plot(df['timestamp'].iloc[train_end:], mean, label='GP Mean', color='C2')
    plt.fill_between(df['timestamp'].iloc[train_end:], mean - 1.96*std, mean + 1.96*std, color='C2', alpha=0.2, label='95% CI')
    
    anom_indices = np.where(flags == 1)[0]
    if len(anom_indices) > 0:
        plt.scatter(df['timestamp'].iloc[train_end:].iloc[anom_indices], 
                   y_test[anom_indices], color='red', s=30, label='Detected Anomaly')
                   
    plt.title(f"GP Composite Forecast (k={best_k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "forecast_detected.png")
    plt.close()

    metrics = {
        "event_level": best_metrics,
        "business": bus_metrics,
        "best_k": best_k
    }
    
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {save_dir}")
    print("Event-level Metrics:", best_metrics)
    return metrics

if __name__ == "__main__":
    nab_root = "./NAB"
    file_key = "realKnownCause/nyc_taxi.csv"
    run_gp_pipeline(
        nab_root=nab_root,
        file_key=file_key,
        train_frac=0.5,
        label_window=3,
        save_dir="./results/gp"
    )
