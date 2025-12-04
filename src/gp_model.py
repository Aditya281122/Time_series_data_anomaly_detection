# src/gp_model.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C
import pandas as pd
from typing import Dict, Tuple

def fit_gp_forecast(y_train: np.ndarray, X_train: np.ndarray = None) -> GaussianProcessRegressor:
    """
    Fit a Gaussian Process Regressor to the training data.
    """
    n = len(y_train)
    # If X_train is not provided, use simple integer indices
    X = np.arange(n).reshape(-1, 1) if X_train is None else X_train
    
    # Kernel: Constant * RBF (smooth trend) + WhiteKernel (noise)
    # We can also add ExpSineSquared for seasonality if we want, but let's start simple as per snippet
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=50.0) + WhiteKernel(noise_level=1.0)
    
    # Kernel: Constant * RBF (smooth trend) + ExpSineSquared (daily seasonality) + WhiteKernel (noise)
    # NYC Taxi data has daily seasonality (48 samples)
    # We combine RBF (for trend/decay) and ExpSineSquared (for seasonality)
    
    long_term_trend = C(50.0) * RBF(length_scale=50.0)
    seasonal_pattern = C(1.0) * ExpSineSquared(length_scale=1.0, periodicity=48.0, periodicity_bounds=(47, 49))
    noise = WhiteKernel(noise_level=1.0)
    
    kernel = long_term_trend + seasonal_pattern + noise
    
    # Normalize y to help with scaling
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y_train)
    return gp

def predict_gp(gp: GaussianProcessRegressor, n_total: int, train_end: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict for test indices [train_end .. n_total-1].
    Returns (mean, std).
    """
    X_test = np.arange(train_end, n_total).reshape(-1, 1)
    mean, std = gp.predict(X_test, return_std=True)
    return mean, std

def plot_gp_forecast(df: pd.DataFrame, train_end_idx: int, mean: np.ndarray, std: np.ndarray, flags: np.ndarray = None, title: str = None, savepath: str = None):
    """
    Plot GP forecast with 95% confidence intervals (1.96 * std).
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    
    plt.figure(figsize=(14, 4))
    plt.plot(df['timestamp'], df['value'], label='actual', alpha=0.6)
    
    test_timestamps = df['timestamp'].iloc[train_end_idx: train_end_idx + len(mean)].reset_index(drop=True)
    plt.plot(test_timestamps, mean, label='GP mean', color='C2')
    plt.fill_between(test_timestamps, lower, upper, color='C2', alpha=0.2, label='95% CI')
    
    if flags is not None:
        anom_times = test_timestamps[flags.astype(bool)]
        if len(anom_times) > 0:
            plt.scatter(anom_times, df.loc[df['timestamp'].isin(anom_times), 'value'], color='red', s=25, label='detected anomaly')
            
    plt.axvline(df['timestamp'].iloc[train_end_idx], color='k', linestyle='--', alpha=0.4, label='prediction start')
    plt.title(title or 'Gaussian Process Forecast')
    plt.legend()
    plt.tight_layout()
    
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150)
    # plt.close()
