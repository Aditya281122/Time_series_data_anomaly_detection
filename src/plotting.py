import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path

def plot_residual_diagnostics(residuals: np.ndarray, title_prefix: str, save_dir: Path):
    """
    Generate and save residual diagnostic plots:
    1. QQ Plot
    2. Histogram with Kurtosis
    3. Rolling Volatility
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. QQ Plot
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{title_prefix} - Residual QQ Plot")
    plt.tight_layout()
    plt.savefig(save_dir / "residual_qq.png")
    plt.close()
    
    # 2. Histogram
    plt.figure(figsize=(10, 6))
    kurt = stats.kurtosis(residuals)
    plt.hist(residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Overlay normal fit
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.title(f"{title_prefix} - Residual Hist (Kurtosis={kurt:.2f})")
    plt.tight_layout()
    plt.savefig(save_dir / "residual_hist.png")
    plt.close()
    
    # 3. Rolling Volatility
    plt.figure(figsize=(12, 6))
    roll_std = pd.Series(residuals).rolling(window=48).std()
    plt.plot(roll_std, color='orange', label='Rolling Std (48h)')
    plt.title(f"{title_prefix} - Rolling Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "residual_rolling_std.png")
    plt.close()

def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, title_prefix: str, save_dir: Path):
    """
    Plot Precision-Recall Curve.
    y_scores should be the raw anomaly score (e.g. z-score or probability).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle NaN in scores
    mask = ~np.isnan(y_scores)
    y_true = y_true[mask]
    y_scores = y_scores[mask]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title_prefix} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curve.png")
    plt.close()
