import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.bsts_model import fit_bsts, predict_bsts, plot_bsts_forecast
from src.utils import rolling_sigma_threshold
from src.load_nab import load_series, load_labels, mark_anomaly_windows
from src.evaluate import compute_detection_metrics, compute_event_level_metrics, compute_business_metrics, persist_filter
from src.plotting import plot_residual_diagnostics, plot_pr_curve

def run_bsts_pipeline(nab_root: str, file_key: str, train_frac: float = 0.5, label_window: int = 1, save_dir: str = "./results/bsts", seasonal_periods: list = [48, 336]):
    """
    Enhanced BSTS pipeline: Daily (48) + Weekly (336) Seasonality.
    Uses Validation Split for Threshold Tuning.
    """
    save_dir = Path(save_dir) / file_key.replace("/", "__")
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    print(f"Loading data: {file_key}")
    df = load_series(nab_root, file_key)
    labels = load_labels(nab_root)
    label_times = labels.get(file_key, labels.get("data/" + file_key, []))
    df = mark_anomaly_windows(df, label_times, window_size=label_window)
    
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + 0.25)) # Use 25% for validation
    
    print(f"Splits: Train [0:{train_end}], Val [{train_end}:{val_end}], Test [{val_end}:{n}]")
    
    y_train = df['value'].iloc[:train_end].values
    
    # Fit on Train
    print(f"--- Training Enhanced BSTS (Seasonality: {seasonal_periods}) ---")
    res = fit_bsts(y_train, seasonal_periods=seasonal_periods)
    
    print("--- Forecasting (Val + Test) ---")
    # Forecast on Val + Test (indices train_end to n-1)
    pred = predict_bsts(res, start=train_end, end=n-1, alpha=0.05, use_dynamic=True)
    mean = pred['mean']
    lower = pred['lower']
    upper = pred['upper']
    
    # Full residuals for Val + Test
    y_val_test = df['value'].iloc[train_end:].values
    residuals_full = y_val_test - mean
    
    # Center residuals (Robust against model bias/collapse)
    residuals_full = residuals_full - np.median(residuals_full)
    
    # Split residuals into Val and Test
    val_len = val_end - train_end
    residuals_val = residuals_full[:val_len]
    residuals_test = residuals_full[val_len:]
    
    y_val_labels = df['is_anomaly'].iloc[train_end:val_end].values
    y_test_labels = df['is_anomaly'].iloc[val_end:].values
    
    # Diagnostics (on Val+Test residuals)
    print("Generating Diagnostics...")
    plot_residual_diagnostics(residuals_full, "BSTS_Residuals", save_dir)
    
    print("--- Tuning Threshold on Validation Set ---")
    
    best_k = 3.0
    best_val_f1 = -1.0
    
    thresholds = np.linspace(2.0, 10.0, 17)
    
    for k_candidate in thresholds:
        # Check on Validation
        flags_val = rolling_sigma_threshold(residuals_val, k=k_candidate, window=48)
        flags_val = persist_filter(flags_val, p=2)
        
        m_val = compute_event_level_metrics(y_val_labels, flags_val, gap=3)
        f1 = m_val['f1']
        
        # Constraint: Anomaly Rate should not be excessive (> 10%)
        # This prevents the "flag everything" solution which hacks the Event-F1 metric
        anomaly_rate = np.mean(flags_val)
        if anomaly_rate > 0.10:
            continue
            
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_k = k_candidate
            
    print(f"--- Best Validation Threshold: k={best_k:.1f} (Val F1={best_val_f1:.4f}) ---\n")
    
    print("--- Evaluating on Test Set ---")
    # Apply best_k to Test
    flags_test = rolling_sigma_threshold(residuals_test, k=best_k, window=48)
    flags_test = persist_filter(flags_test, p=2)
    
    # Metrics on Test
    metrics_test = compute_event_level_metrics(y_test_labels, flags_test, gap=3)
    
    # PR Curve (on Test)
    roll_sigma_test = pd.Series(residuals_test).rolling(window=48, min_periods=1).std().fillna(method='bfill').values
    anomaly_scores_test = np.abs(residuals_test) / (roll_sigma_test + 1e-9)
    plot_pr_curve(y_test_labels, anomaly_scores_test, "BSTS_Test_PR_Curve", save_dir)
    
    # Business Metrics (on Test)
    bus_metrics = compute_business_metrics(
        y_test_labels, 
        flags_test, 
        df['timestamp'].iloc[val_end:], 
        gap=3
    )
    print(f"Test FP/day: {bus_metrics['fp_per_day']:.2f}, Latency: {bus_metrics['median_latency_minutes']:.1f} min")
    
    # Save results (Full Val+Test for visualization)
    # We need to concatenate flags_val and flags_test for the full output dataframe
    # Re-compute flags for Val using best_k for consistency in output
    flags_val_final = rolling_sigma_threshold(residuals_val, k=best_k, window=48)
    flags_val_final = persist_filter(flags_val_final, p=2)
    
    flags_full = np.concatenate([flags_val_final, flags_test])
    
    out_df = df.iloc[train_end:].reset_index(drop=True).copy()
    out_df['bsts_mean'] = mean
    out_df['bsts_lower'] = lower
    out_df['bsts_upper'] = upper
    out_df['detected'] = flags_full
    out_df['split'] = ['Validation'] * val_len + ['Test'] * (len(out_df) - val_len)
    out_df.to_csv(save_dir / "predictions.csv", index=False)
    
    metrics = {
        "event_level": metrics_test,
        "business": bus_metrics,
        "best_k": best_k,
        "val_f1": best_val_f1
    }
    
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved results to {save_dir}")
    print("Test Event Metrics:", metrics_test)
    
    # Plot Forecast
    plot_path = str(save_dir / "forecast_detected.png")
    plot_bsts_forecast(df, train_end, mean, lower, upper, flags_full, title=f"BSTS (Daily+Weekly): {file_key}", savepath=plot_path)

if __name__ == "__main__":
    import argparse
    
    # Robustly determine project root and NAB path
    current_dir = Path(os.getcwd())
    if current_dir.name == 'src':
        project_root = current_dir.parent
    else:
        project_root = current_dir

    nab_root = str(project_root / 'NAB')
    file_key = "realKnownCause/nyc_taxi.csv"
    
    run_bsts_pipeline(
        nab_root=nab_root,
        file_key=file_key,
        train_frac=0.5,
        label_window=3,
        save_dir="./results/bsts"
    )
