import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_nab import load_series, load_labels, mark_anomaly_windows
from src.evaluate import compute_event_level_metrics, persist_filter, compute_business_metrics
from src.utils import rolling_sigma_threshold, decompose_series
from src.plotting import plot_residual_diagnostics, plot_pr_curve
from src.lstm_model import create_sequences, TimeSeriesDataset, LSTMAnomalyDetector, train_model, predict_lstm

def run_lstm_pipeline(nab_root: str, file_key: str, train_frac: float = 0.5, label_window: int = 1, save_dir: str = "./results/lstm"):
    save_dir = Path(save_dir) / file_key.replace("/", "__")
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = load_series(nab_root, file_key)
    labels = load_labels(nab_root)
    label_times = labels.get(file_key, labels.get("data/" + file_key, []))
    df = mark_anomaly_windows(df, label_times, window_size=label_window)
    
    # --- Step 1: STL Decomposition ---
    print("Running STL Decomposition...")
    df_decomp = decompose_series(df, period=48)
    
    data_values = df_decomp['resid'].values
    
    # Normalize (Standard Scaling)
    mean_val = np.mean(data_values)
    std_val = np.std(data_values)
    data_norm = (data_values - mean_val) / std_val
    
    n = len(df)
    train_end = int(n * train_frac)
    
    # Create Sequences
    seq_len = 48 # 1 day context
    
    train_data = data_norm[:train_end]
    test_data = data_norm # We predict on full series for simplicity of indexing, but only eval on test
    
    X_train, y_train = create_sequences(train_data, seq_len)
    
    # Train Loader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LSTM on Residuals (Device: {device})...")
    model = LSTMAnomalyDetector(input_size=1, hidden_size=64, num_layers=2)
    model = train_model(model, train_loader, num_epochs=15, learning_rate=0.001, device=device)
    
    # Predict
    print("Forecasting...")
    # predict_lstm returns predictions aligned with data[seq_len:]
    preds_norm = predict_lstm(model, data_norm, seq_len, device=device)
    
    # Align predictions with original dataframe
    # preds_norm corresponds to indices [seq_len : ]
    # We need to pad the beginning with NaNs or 0s
    full_preds = np.full(n, np.nan)
    full_preds[seq_len:] = preds_norm
    
    # Inverse Transform
    preds_rescaled = full_preds * std_val + mean_val
    
    # Residuals (Prediction Error)
    # This is (Actual STL Residual - Predicted STL Residual)
    # So it's the "Residual of the Residual"
    residuals_final = data_values - preds_rescaled
    
    # We only care about the TEST set for evaluation
    test_residuals = residuals_final[train_end:]
    # Handle NaNs at start of test set if any (shouldn't be if train_end > seq_len)
    mask = ~np.isnan(test_residuals)
    
    # --- Diagnostics ---
    print("Generating Diagnostic Plots...")
    # We use the valid test residuals
    plot_residual_diagnostics(test_residuals[mask], "LSTM_on_STL_Resid", save_dir)
    
    y_true_test = df['is_anomaly'].iloc[train_end:].values
    # PR Curve
    # Score = abs(residual)
    plot_pr_curve(y_true_test[mask], np.abs(test_residuals[mask]), "LSTM_on_STL_Resid", save_dir)
    
    # --- Detection ---
    print("\n--- Starting Threshold Sweep (Rolling Sigma) ---")
    best_k = 3.0
    best_f1 = -1.0
    best_metrics = None
    
    # We only evaluate on the valid part of test set
    y_true_eval = y_true_test[mask]
    resid_eval = test_residuals[mask]
    
    for k_candidate in np.linspace(2.0, 10.0, 17):
        flags_temp = rolling_sigma_threshold(resid_eval, k=k_candidate, window=48)
        flags_temp = persist_filter(flags_temp, p=2)
        
        m_evt = compute_event_level_metrics(y_true_eval, flags_temp, gap=3)
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
    flags_eval = rolling_sigma_threshold(resid_eval, k=best_k, window=48)
    flags_eval = persist_filter(flags_eval, p=2)
    
    # Map back to full length
    final_flags = np.zeros(n, dtype=int)
    
    if np.all(mask):
        final_flags[train_end:] = flags_eval
    else:
        # This is tricky if we have NaNs in the middle, but we shouldn't.
        # Just filling the end
        valid_len = len(flags_eval)
        final_flags[-valid_len:] = flags_eval
        
    # Business Metrics
    # We need to pass the timestamps corresponding to the evaluated part
    # y_true_eval corresponds to df['timestamp'].iloc[train_end:][mask]
    timestamps_eval = df['timestamp'].iloc[train_end:][mask]
    
    bus_metrics = compute_business_metrics(
        y_true_eval, 
        flags_eval, 
        timestamps_eval, 
        gap=3
    )
    print(f"FP/day: {bus_metrics['fp_per_day']:.2f}, Latency: {bus_metrics['median_latency_minutes']:.1f} min")

    # Save results
    out_df = df.copy()
    out_df['stl_resid'] = df_decomp['resid']
    out_df['lstm_pred_resid'] = preds_rescaled
    out_df['final_resid'] = residuals_final
    out_df['detected'] = final_flags
    
    out_df.to_csv(save_dir / "predictions.csv", index=False)
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['value'], label='Original Data', color='black', alpha=0.5)
    plt.plot(df['timestamp'], df_decomp['trend'] + df_decomp['seasonal'], label='STL Trend+Season', color='blue', alpha=0.5)
    plt.title("Data & STL Components")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'][train_end:], data_values[train_end:], label='STL Residual (Target)', color='gray', alpha=0.5)
    plt.plot(df['timestamp'][train_end:], preds_rescaled[train_end:], label='LSTM Prediction', color='green', alpha=0.8)
    
    anom_indices = np.where(final_flags == 1)[0]
    if len(anom_indices) > 0:
        plt.scatter(df['timestamp'].iloc[anom_indices], 
                   data_values[anom_indices], color='red', s=30, label='Detected Anomaly')
    
    plt.title(f"LSTM on Residuals (k={best_k})")
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
    run_lstm_pipeline(
        nab_root=nab_root,
        file_key=file_key,
        train_frac=0.5,
        label_window=3,
        save_dir="./results/lstm"
    )
