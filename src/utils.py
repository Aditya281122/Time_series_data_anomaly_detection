import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def decompose_series(df: pd.DataFrame, period: int = 48) -> pd.DataFrame:
    """
    Decompose a time series using STL.
    
    Args:
        df: DataFrame with 'timestamp' and 'value' columns.
        period: Seasonal period in samples (e.g., 48 for 30-min data with daily seasonality).
        
    Returns:
        DataFrame with columns ['value', 'trend', 'seasonal', 'resid'].
    """
    # Ensure value is numeric
    values = df['value'].values
    
    # STL Decomposition
    # robust=True is important for anomaly detection contexts
    stl = STL(values, period=period, robust=True)
    res = stl.fit()
    
    out_df = df.copy()
    out_df['trend'] = res.trend
    out_df['seasonal'] = res.seasonal
    out_df['resid'] = res.resid
    
    return out_df

def mad_sigma(arr: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation (MAD) based sigma estimate.
    sigma ~ 1.4826 * MAD
    """
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return 1.4826 * mad

def rolling_sigma_threshold(
    residuals: np.ndarray, 
    k: float = 3.0, 
    window: int = 48, 
    min_periods: int = 5
) -> np.ndarray:
    """
    Detect anomalies using a rolling standard deviation threshold.
    
    Args:
        residuals: 1D numpy array of residuals (actual - predicted).
        k: Threshold multiplier (number of sigmas).
        window: Rolling window size.
        min_periods: Minimum periods for rolling calculation.
        
    Returns:
        Binary flag array (1 for anomaly, 0 for normal).
    """
    # Convert to Series for rolling
    resid_s = pd.Series(residuals)
    
    # Compute rolling mean and std
    # We use bfill to handle the start of the series
    rolling = resid_s.rolling(window=window, min_periods=min_periods)
    sigma_roll = rolling.std().fillna(method='bfill')
    mean_roll = rolling.mean().fillna(method='bfill')
    
    # Add epsilon to prevent zero threshold
    sigma_roll = sigma_roll + 1e-6
    
    # Check deviation from rolling mean (Adaptive Threshold)
    flags = (np.abs(residuals - mean_roll) > k * sigma_roll).astype(int)
    return flags.values
