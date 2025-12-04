# src/load_nab.py
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict

NAB_ROOT_DEFAULT = Path("./NAB")  # <-- adjust only if your NAB folder is elsewhere

def load_series(nab_root: str = None, file_key: str = None) -> pd.DataFrame:
    """
    Load a single NAB timeseries CSV into a DataFrame.
    nab_root: path to NAB repository root (default ./NAB)
    file_key: relative path inside NAB/data, e.g. "data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv"
              OR the shorter key "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv"
    Returns DataFrame with columns ['timestamp', 'value'].
    """
    root = Path(nab_root or NAB_ROOT_DEFAULT)
    # support both forms of file_key
    if file_key.startswith("data/"):
        csv_path = root / file_key
    else:
        csv_path = root / "data" / file_key

    df = pd.read_csv(csv_path)
    # handle files without headers
    if 'timestamp' not in df.columns or 'value' not in df.columns:
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['timestamp', 'value']
        else:
            raise ValueError(f"Unexpected CSV format: {csv_path}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def load_labels(nab_root: str = None) -> Dict[str, List[str]]:
    """
    Load NAB combined labels file and return a dict mapping file_key -> list of anomaly timestamp strings.
    """
    root = Path(nab_root or NAB_ROOT_DEFAULT)
    labels_path = root / "labels" / "combined_labels.json"
    with open(labels_path, 'r') as fh:
        labels = json.load(fh)
    return labels

def mark_anomaly_windows(df: pd.DataFrame, anomaly_timestamps: List[str], window_size: int = 1) -> pd.DataFrame:
    """
    Mark 'is_anomaly' column in df: 1 if within +/- window_size rows of a labeled timestamp.
    window_size counts rows (samples). Choose window_size based on sampling density.
    """
    df = df.copy()
    df['is_anomaly'] = 0
    if not anomaly_timestamps:
        return df
    anoms = [pd.to_datetime(ts) for ts in anomaly_timestamps]
    for t in anoms:
        idx = df['timestamp'].searchsorted(t)
        lo = max(0, idx - window_size)
        hi = min(len(df) - 1, idx + window_size)
        df.loc[lo:hi, 'is_anomaly'] = 1
    return df
