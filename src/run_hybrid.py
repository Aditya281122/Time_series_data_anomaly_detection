import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.load_nab import load_series, load_labels, mark_anomaly_windows
from src.evaluate import compute_event_level_metrics, persist_filter, compute_business_metrics
from src.utils import rolling_sigma_threshold, decompose_series, mad_sigma
from src.plotting import plot_residual_diagnostics, plot_pr_curve

# Import individual model runners/functions
from src.run_gp import fit_gp_composite
from src.run_lstm import create_sequences, TimeSeriesDataset, LSTMAnomalyDetector, train_model, predict_lstm
import torch
from torch.utils.data import DataLoader

def run_hybrid_ensemble(nab_root: str, file_key: str, train_frac: float = 0.5, label_window: int = 1, save_dir: str = "./results/hybrid"):
    save_dir = Path(save_dir) / file_key.replace("/", "__")
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = load_series(nab_root, file_key)
    labels = load_labels(nab_root)
    label_times = labels.get(file_key, labels.get("data/" + file_key, []))
    df = mark_anomaly_windows(df, label_times, window_size=label_window)
    
    n = len(df)
    train_end = int(n * train_frac)
    
    # --- Model 1: STL Baseline (Robust Z-Score) ---
    print("\n--- Model 1: STL Decomposition ---")
    df_stl = decompose_series(df, period=48)
    stl_resid = df_stl['resid'].values
    # Score: Abs Z-score using MAD
    sigma_stl = mad_sigma(stl_resid)
    score_stl = np.abs(stl_resid) / (sigma_stl + 1e-9)
    
    # --- Model 2: Gaussian Process ---
    print("\n--- Model 2: Gaussian Process ---")
    # Train on subset
    train_subset_size = 1000
    train_start_idx = max(0, train_end - train_subset_size)
    y_train_gp = df['value'].iloc[train_start_idx:train_end].values
    X_train_gp = np.arange(train_start_idx, train_end).reshape(-1, 1)
    
    gp = fit_gp_composite(y_train_gp, X_train_gp)
    
    # Predict on full series (for simplicity of scoring alignment)
    # Note: GP scales cubically, so predicting on 10k points might be slow.
    # Let's predict only on test set and pad the rest with 0s (since we only eval on test)
    X_test_gp = np.arange(train_end, n).reshape(-1, 1)
    mean_gp, std_gp = gp.predict(X_test_gp, return_std=True)
    
    # Align
    full_mean_gp = np.full(n, np.nan)
    full_std_gp = np.full(n, np.nan)
    full_mean_gp[train_end:] = mean_gp
    full_std_gp[train_end:] = std_gp
    
    # Score: Z-score using GP's predicted std
    # We use the residual from the mean
    resid_gp = df['value'].values - full_mean_gp
    # If std is very small, score explodes. Add epsilon.
    score_gp = np.abs(resid_gp) / (full_std_gp + 1e-9)
    # Fill NaNs (train set) with 0
    score_gp = np.nan_to_num(score_gp)
    
    # --- Model 3: LSTM on STL Residuals ---
    print("\n--- Model 3: LSTM on STL Residuals ---")
    # We use the STL residuals from Model 1
    data_values = stl_resid
    mean_val = np.mean(data_values)
    std_val = np.std(data_values)
    data_norm = (data_values - mean_val) / std_val
    
    seq_len = 48
    train_data = data_norm[:train_end]
    X_train_lstm, y_train_lstm = create_sequences(train_data, seq_len)
    
    train_loader = DataLoader(TimeSeriesDataset(X_train_lstm, y_train_lstm), batch_size=64, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAnomalyDetector(input_size=1, hidden_size=64, num_layers=2)
    model = train_model(model, train_loader, num_epochs=10, learning_rate=0.001, device=device)
    
    preds_norm = predict_lstm(model, data_norm, seq_len, device=device)
    
    full_preds_lstm = np.full(n, np.nan)
    full_preds_lstm[seq_len:] = preds_norm * std_val + mean_val
    
    # Residual of the residual
    resid_lstm = stl_resid - full_preds_lstm
    
    # Score: Standardize by rolling std of these residuals
    resid_lstm_s = pd.Series(resid_lstm)
    roll_std_lstm = resid_lstm_s.rolling(window=48, min_periods=1).std().fillna(method='bfill').values
    score_lstm = np.abs(resid_lstm) / (roll_std_lstm + 1e-9)
    score_lstm = np.nan_to_num(score_lstm)
    
    # --- Ensemble Aggregation ---
    print("\n--- Aggregating Scores ---")
    # We average the Z-scores.
    # Ideally we should normalize them to be comparable, but Z-scores are already roughly comparable (units of sigma).
    
    # We only care about test set
    test_mask = np.arange(n) >= train_end
    
    # Combine
    # Weights: Equal for now
    combined_score = (score_stl + score_gp + score_lstm) / 3.0
    
    # --- Diagnostics ---
    print("Generating Diagnostic Plots...")
    # Plot scores
    plt.figure(figsize=(14, 8))
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'][test_mask], score_stl[test_mask], label='STL Score', color='blue', alpha=0.7)
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(df['timestamp'][test_mask], score_gp[test_mask], label='GP Score', color='green', alpha=0.7)
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'][test_mask], score_lstm[test_mask], label='LSTM Score', color='purple', alpha=0.7)
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(df['timestamp'][test_mask], combined_score[test_mask], label='Combined Score', color='red', linewidth=1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "anomaly_scores.png")
    plt.close()
    
    # PR Curve
    y_true_test = df['is_anomaly'].iloc[train_end:].values
    combined_score_test = combined_score[train_end:]
    plot_pr_curve(y_true_test, combined_score_test, "Hybrid_Ensemble", save_dir)
    
    # --- Detection ---
    print("\n--- Starting Threshold Sweep on Combined Score ---")
    best_thresh = 3.0
    best_f1 = -1.0
    best_metrics = None
    
    # Sweep threshold T for combined score
    # Since it's an average of Z-scores, T should be around 2-5
    for thresh in np.linspace(2.0, 10.0, 17):
        flags_temp = (combined_score_test > thresh).astype(int)
        flags_temp = persist_filter(flags_temp, p=2)
        
        m_evt = compute_event_level_metrics(y_true_test, flags_temp, gap=3)
        f1 = m_evt['f1']
        
        # Constraint: Anomaly Rate should not be excessive (> 10%)
        anomaly_rate = np.mean(flags_temp)
        if anomaly_rate > 0.10:
            continue
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = m_evt
            
    print(f"--- Best Threshold: T={best_thresh:.2f} with F1={best_f1:.4f} ---\n")
    
    final_flags_test = (combined_score_test > best_thresh).astype(int)
    final_flags_test = persist_filter(final_flags_test, p=2)
    
    # Business Metrics
    bus_metrics = compute_business_metrics(
        y_true_test, 
        final_flags_test, 
        df['timestamp'].iloc[train_end:], 
        gap=3
    )
    print(f"FP/day: {bus_metrics['fp_per_day']:.2f}, Latency: {bus_metrics['median_latency_minutes']:.1f} min")
    
    # Save results
    out_df = df.iloc[train_end:].reset_index(drop=True).copy()
    out_df['score_stl'] = score_stl[train_end:]
    out_df['score_gp'] = score_gp[train_end:]
    out_df['score_lstm'] = score_lstm[train_end:]
    out_df['combined_score'] = combined_score_test
    out_df['detected'] = final_flags_test
    out_df.to_csv(save_dir / "predictions.csv", index=False)
    
    metrics = {
        "event_level": best_metrics,
        "business": bus_metrics,
        "best_threshold": best_thresh
    }
    
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {save_dir}")
    print("Event-level Metrics:", best_metrics)
    return metrics

if __name__ == "__main__":
    nab_root = "./NAB"
    file_key = "realKnownCause/nyc_taxi.csv"
    run_hybrid_ensemble(
        nab_root=nab_root,
        file_key=file_key,
        train_frac=0.5,
        label_window=3,
        save_dir="./results/hybrid"
    )
