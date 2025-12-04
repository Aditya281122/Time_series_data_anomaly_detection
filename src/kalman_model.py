# src/kalman_model.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def fit_kalman_local_level(y_train: np.ndarray):
    """
    Fit a local-level UnobservedComponents model (Kalman-style).
    y_train: 1D numpy array (training portion of series)
    Returns: fitted results object (statsmodels)
    """
    model = sm.tsa.UnobservedComponents(endog=y_train, level='local level')
    res = model.fit(disp=False)
    return res

def predict_with_intervals(res, start: int, end: int, alpha: float = 0.05, use_dynamic: bool = False) -> Dict[str, np.ndarray]:
    """
    Predict with confidence intervals for indices start..end (inclusive).

    If use_dynamic=True, statisticsmodels will do 1-step-ahead predictions
    from 'start' onward, updating the state as it sees new observations.
    This is what we want for anomaly detection.
    """
    if use_dynamic:
        pred = res.get_prediction(start=start, end=end, dynamic=True)
    else:
        pred = res.get_prediction(start=start, end=end)

    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=alpha)
    
    # ci is likely a numpy array because we passed numpy array to fit()
    if isinstance(ci, pd.DataFrame):
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
    else:
        lower = ci[:, 0]
        upper = ci[:, 1]
        
    return {"mean": mean, "lower": lower, "upper": upper}




def detect_anomalies_from_intervals(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Simple binary anomaly flag: 1 when actual < lower or actual > upper.
    """
    flags = ((actual < lower) | (actual > upper)).astype(int)
    return flags

def detect_anomalies_by_residual(actual: np.ndarray, mean: np.ndarray, train_residuals: np.ndarray, k: float = 3.0, use_mad: bool = False, persistence: int = 0, use_rolling: bool = False, window: int = 50) -> np.ndarray:
    """
    Statistical anomaly detection based on residual z-score.

    - train_residuals: residuals on the training window (res.resid)
    - k: threshold in std deviations (e.g. 3.0 for ~3σ rule)
    - use_mad: if True, use Median Absolute Deviation (robust sigma) instead of std
    - persistence: if > 0, require 'persistence' consecutive flags to count as anomaly
    - use_rolling: if True, use rolling standard deviation of TEST residuals (adaptive threshold)
    - window: window size for rolling sigma
    
    Returns binary flags: 1 if |actual - mean| > k * sigma else 0.
    """
    residuals = actual - mean
    
    if use_rolling:
        # Local rolling sigma on test residuals
        # We use bfill to handle the start of the window
        resid_series = pd.Series(residuals)
        sigma = resid_series.rolling(window=window, min_periods=5).std().fillna(method='bfill').values
        # Ensure no zero sigma
        sigma[sigma == 0] = np.std(train_residuals) if len(train_residuals) > 0 else 1.0
    elif use_mad:
        # MAD = median(|x - median(x)|)
        # sigma_mad = 1.4826 * MAD (for consistency with normal distribution)
        mad = np.median(np.abs(train_residuals - np.median(train_residuals)))
        sigma = 1.4826 * mad
    else:
        sigma = np.std(train_residuals, ddof=1)
        
    # Handle scalar sigma case
    if np.ndim(sigma) == 0:
        if sigma == 0 or np.isnan(sigma):
            return np.zeros_like(actual, dtype=int)
    
    z = np.abs(residuals) / sigma
    flags = (z > k).astype(int)
    
    if persistence > 1:
        # filter out short bursts
        flags = apply_persistence_filter(flags, persistence)
        
    return flags

def apply_persistence_filter(flags: np.ndarray, p: int) -> np.ndarray:
    """
    Remove sequences of 1s shorter than p.
    """
    out = flags.copy()
    n = len(flags)
    i = 0
    while i < n:
        if flags[i] == 1:
            j = i
            while j + 1 < n and flags[j+1] == 1:
                j += 1
            # run length is j - i + 1
            if (j - i + 1) < p:
                out[i:j+1] = 0
            i = j + 1
        else:
            i += 1
    return out

def plot_forecast(df: pd.DataFrame, train_end_idx: int, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray, flags: np.ndarray = None, title: str = None, savepath: str = None):
    """
    Plot actual series, predicted mean and CI for test portion.
    train_end_idx: index of last training sample (prediction starts from train_end_idx)
    mean/lower/upper: numpy arrays for predicted steps
    flags: optional binary array of same length as mean indicating detected anomalies
    """
    plt.figure(figsize=(14,4))
    plt.plot(df['timestamp'], df['value'], label='actual', alpha=0.6)
    test_timestamps = df['timestamp'].iloc[train_end_idx: train_end_idx + len(mean)].reset_index(drop=True)
    plt.plot(test_timestamps, mean, label='pred_mean', color='C1')
    plt.fill_between(test_timestamps, lower, upper, color='C1', alpha=0.2, label='95% CI')
    if flags is not None:
        # mark detected anomalies on the plot
        anom_times = test_timestamps[flags.astype(bool)]
        anom_vals = df['value'].iloc[test_timestamps.index[0] + flags.nonzero()[0]] if len(flags.nonzero()[0])>0 else []
        plt.scatter(anom_times, df.loc[df['timestamp'].isin(anom_times), 'value'], color='red', s=25, label='detected anomaly')
    plt.axvline(df['timestamp'].iloc[train_end_idx], color='k', linestyle='--', alpha=0.4, label='prediction start')
    plt.title(title or 'Kalman Forecast + detected anomalies')
    plt.legend()
    plt.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150)
    # plt.show() # blocking call

def run_kalman_pipeline(nab_root: str, file_key: str, train_frac: float = 0.7, label_window: int = 1, save_dir: str = "./results/kalman"):
    """
    End-to-end: load series, fit kalman, forecast, detect anomalies, save plot & csv.
    nab_root: path to NAB folder
    file_key: e.g. "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv"
    """
    from src.load_nab import load_series, load_labels, mark_anomaly_windows
    import json
    save_dir = Path(save_dir) / file_key.replace("/", "__")
    save_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = load_series(nab_root, file_key)
    labels = load_labels(nab_root)
    label_times = labels.get(file_key, labels.get("data/" + file_key, []))
    df = mark_anomaly_windows(df, label_times, window_size=label_window)
    total_anoms = int(df['is_anomaly'].sum())
    n = len(df)
    train_end = int(n * train_frac)
    test_anoms = int(df['is_anomaly'].iloc[train_end:].sum())
    print(f"Total labeled anomalies in series: {total_anoms}")
    print(f"Labeled anomalies in TEST region: {test_anoms} (train_frac={train_frac})")

    n = len(df)
    train_end = int(n * train_frac)
    y_train = df['value'].iloc[:train_end].values
    y_test = df['value'].iloc[train_end:].values

    # fit
    res = fit_kalman_local_level(y_train)

        # forecast (predict test range)
    pred = predict_with_intervals(res, start=train_end, end=n-1, alpha=0.05)
    mean = pred['mean']
    # We'll still plot the model's CI, but detection uses residual statistics
    lower_model = pred['lower']
    upper_model = pred['upper']

    # compute residual-based anomalies
    train_residuals = res.resid  # residuals on training data
    # k controls how aggressive we are: start with 3.0, then tune (2.5, 2.0, etc.)
    flags = detect_anomalies_by_residual(y_test, mean, train_residuals, k=3.0)

    # For plotting, build a symmetric band using residual sigma
    sigma_train = np.std(train_residuals, ddof=1)
    lower = mean - 3.0 * sigma_train
    upper = mean + 3.0 * sigma_train


    # detect
    # detect
    # flags = detect_anomalies_from_intervals(y_test, lower, upper)
    
    # Use the new residual-based detection (3-sigma rule)
    # We need training residuals to compute sigma
    train_residuals = res.resid
    
    # Use MAD and Persistence (k=3.0, persistence=2)
    flags = detect_anomalies_by_residual(
        y_test, 
        mean, 
        train_residuals, 
        k=3.0, 
        use_mad=True, 
        persistence=2
    )

    # evaluation will be done by evaluate.py
    # save results
    out_df = df.iloc[train_end:].reset_index(drop=True).copy()
    out_df['pred_mean'] = mean
    out_df['pred_lower'] = lower
    out_df['pred_upper'] = upper
    out_df['detected'] = flags
    out_df.to_csv(save_dir / "predictions.csv", index=False)

    # plot
    plot_path = str(save_dir / "forecast_detected.png")
    plot_forecast(df, train_end, mean, lower, upper, flags, title=file_key, savepath=plot_path)

    # compute metrics
    from src.evaluate import compute_detection_metrics, compute_event_level_metrics
    
    # Pointwise metrics
    metrics_pointwise = compute_detection_metrics(df['is_anomaly'].iloc[train_end:].values, flags)
    
    # Event-level metrics
    metrics_event = compute_event_level_metrics(df['is_anomaly'].iloc[train_end:].values, flags)
    
    metrics = {
        "pointwise": metrics_pointwise,
        "event_level": metrics_event
    }
    
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved results to {save_dir}")
    print("Pointwise Metrics:", metrics_pointwise)
    print("Event-level Metrics:", metrics_event)
    return metrics
