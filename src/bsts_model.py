# src/bsts_model.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# Reuse detection logic from kalman_model to ensure consistency
from src.kalman_model import detect_anomalies_by_residual

def fit_bsts(y_train: np.ndarray, seasonal_period: int = None, seasonal_periods: list = None):
    """
    Fit a Bayesian Structural Time Series model using UnobservedComponents.
    
    Components:
    - Level: Local Linear Trend (level + slope)
    - Seasonality: Stochastic seasonal component (if seasonal_period is provided)
    - Multiple Seasonalities: If seasonal_periods list is provided, use freq_seasonal (trigonometric).
    """
    # Define model components
    # 'local linear trend' adds a stochastic trend (level + slope)
    level_type = 'local linear trend'
    
    if seasonal_periods:
        # Multiple seasonalities using trigonometric representation
        # Limit harmonics to avoid massive state space for long periods (e.g. weekly=336)
        # 10 harmonics is usually enough to capture the main shape
        freq_seasonal = [{'period': p, 'harmonics': min(int(p/2), 10)} for p in seasonal_periods]
        
        model = sm.tsa.UnobservedComponents(
            endog=y_train, 
            level=level_type,
            freq_seasonal=freq_seasonal
        )
    elif seasonal_period:
        # Add a stochastic seasonal component (dummy variable approach)
        model = sm.tsa.UnobservedComponents(
            endog=y_train, 
            level=level_type,
            seasonal=seasonal_period
        )
    else:
        model = sm.tsa.UnobservedComponents(
            endog=y_train, 
            level=level_type
        )
        
    res = model.fit(disp=False)
    return res

def predict_bsts(res, start: int, end: int, alpha: float = 0.05, use_dynamic: bool = False) -> Dict[str, np.ndarray]:
    """
    Predict with confidence intervals.
    """
    if use_dynamic:
        pred = res.get_prediction(start=start, end=end, dynamic=True)
    else:
        pred = res.get_prediction(start=start, end=end)

    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=alpha)
    
    if isinstance(ci, pd.DataFrame):
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
    else:
        lower = ci[:, 0]
        upper = ci[:, 1]
        
    return {"mean": mean, "lower": lower, "upper": upper}

def plot_bsts_forecast(df: pd.DataFrame, train_end_idx: int, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray, flags: np.ndarray = None, title: str = None, savepath: str = None):
    """
    Plot BSTS forecast and anomalies.
    """
    plt.figure(figsize=(14,4))
    plt.plot(df['timestamp'], df['value'], label='actual', alpha=0.6)
    
    test_timestamps = df['timestamp'].iloc[train_end_idx: train_end_idx + len(mean)].reset_index(drop=True)
    plt.plot(test_timestamps, mean, label='BSTS mean', color='C2') # Use different color for BSTS
    plt.fill_between(test_timestamps, lower, upper, color='C2', alpha=0.2, label='95% CI')
    
    if flags is not None:
        anom_times = test_timestamps[flags.astype(bool)]
        plt.scatter(anom_times, df.loc[df['timestamp'].isin(anom_times), 'value'], color='red', s=25, label='detected anomaly')
        
    plt.axvline(df['timestamp'].iloc[train_end_idx], color='k', linestyle='--', alpha=0.4, label='prediction start')
    plt.title(title or 'BSTS Forecast + detected anomalies')
    plt.legend()
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150)
    # plt.show()
