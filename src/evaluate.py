# src/evaluate.py
import numpy as np
import pandas as pd
from typing import Dict

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute precision, recall, f1 for binary arrays.
    """
    eps = 1e-9
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def compute_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true / y_pred should be 1D numpy arrays of same length
    Returns precision, recall, f1 and detection rate (TPR)
    """
    y_pred = y_pred.astype(int)
    return precision_recall_f1(y_true, y_pred)

def merge_flags_to_events(flags, gap=1):
    """
    Collapse consecutive 1s into event tuples (start_idx, end_idx).
    end_idx is inclusive.
    Merges events that are <= gap samples apart.
    """
    events = []
    n = len(flags)
    i = 0
    while i < n:
        if flags[i] == 1:
            j = i
            while j + 1 < n and flags[j+1] == 1:
                j += 1
            events.append([i, j])
            i = j + 1
        else:
            i += 1
            
    # merge events that are <= gap apart
    if gap > 0 and len(events) > 1:
        merged = []
        for s, e in events:
            if not merged:
                merged.append([s, e])
            else:
                # if start of current is <= end of last + gap + 1
                # e.g. last=[0,1], curr=[3,4], gap=1. 3 - 1 = 2. 2 > 1? Yes. No merge.
                # gap is number of zeros allowed between ones.
                # if gap=1, 1 0 1 -> merged. indices 0, 2. 2 - 0 = 2. gap+1 = 2.
                if s - merged[-1][1] <= gap + 1:
                    merged[-1][1] = e
                else:
                    merged.append([s, e])
        return [tuple(x) for x in merged]
        
    return [tuple(x) for x in events]

def compute_event_level_metrics(y_true: np.ndarray, y_pred: np.ndarray, gap: int = 0) -> Dict[str, float]:
    """
    Compute precision/recall/f1 based on EVENT OVERLAP.
    A predicted event is a TP if it overlaps with ANY true event.
    
    gap: max number of consecutive zeros to bridge within an event.
    """
    true_events = merge_flags_to_events(y_true, gap=gap)
    pred_events = merge_flags_to_events(y_pred, gap=gap)
    
    tp = 0
    # TP: A predicted event that overlaps with at least one true event
    for ps, pe in pred_events:
        if any(not (pe < ts or ps > te) for ts, te in true_events):
            tp += 1
            
    fp = len(pred_events) - tp
    
    # FN: A true event that is NOT overlapped by any predicted event
    # Note: The user's snippet used `fn = len(true_events) - tp`.
    # This assumes 1-to-1 mapping or that we only care about "did we predict something valid".
    # Standard definition usually checks if true events were missed.
    # If multiple preds overlap one true, tp increases (in user logic).
    # If one pred overlaps multiple trues, tp increases by 1.
    # Let's stick to the user's logic:
    # "Score at event-level (TP if predicted event overlaps labeled event)"
    # This implies precision-focused TP.
    # But for Recall, we need to know how many TRUE events were hit.
    
    # Let's refine TP for Recall:
    tp_for_recall = 0
    for ts, te in true_events:
        if any(not (pe < ts or ps > te) for ps, pe in pred_events):
            tp_for_recall += 1
            
    fn = len(true_events) - tp_for_recall
    
    # User's snippet:
    # prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    # This uses the SAME tp for both. This is a simplification where 
    # we assume roughly 1-to-1 or just count "valid predictions".
    # However, to be strictly correct:
    # Precision = (Predicted events that are real) / Total Predicted
    # Recall = (Real events that were detected) / Total Real
    
    precision = tp / (len(pred_events) + 1e-9)
    recall = tp_for_recall / (len(true_events) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "tp": tp, 
        "fp": fp, 
        "fn": fn,
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events)
    }

def persist_filter(flags: np.ndarray, p: int = 2) -> np.ndarray:
    """
    Remove anomalies that don't persist for at least p consecutive steps.
    """
    if p <= 1:
        return flags
        
    out = flags.copy()
    n = len(flags)
    i = 0
    while i < n:
        if flags[i] == 1:
            j = i
            while j+1 < n and flags[j+1] == 1:
                j += 1
            # Length of sequence is j - i + 1
            if (j - i + 1) < p:
                out[i:j+1] = 0
            i = j + 1
        else:
            i += 1
    return out

def compute_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.Series, gap: int = 0) -> Dict[str, float]:
    """
    Compute business-relevant metrics:
    - FP_per_day: False Positives per day.
    - Median_Latency: Median delay in detection (in minutes).
    """
    true_events = merge_flags_to_events(y_true, gap=gap)
    pred_events = merge_flags_to_events(y_pred, gap=gap)
    
    # FP Calculation
    # We use the event-level FP count from before
    tp = 0
    for ps, pe in pred_events:
        if any(not (pe < ts or ps > te) for ts, te in true_events):
            tp += 1
    fp_count = len(pred_events) - tp
    
    # Duration in days
    if len(timestamps) > 1:
        # Assuming timestamps are datetime objects
        duration_days = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (3600 * 24)
    else:
        duration_days = 1.0
        
    fp_per_day = fp_count / duration_days if duration_days > 0 else 0.0
    
    # Latency Calculation
    # For each True Event, find the FIRST predicted event that overlaps it
    latencies = []
    for ts, te in true_events:
        # Find overlapping predictions
        overlaps = []
        for ps, pe in pred_events:
            if not (pe < ts or ps > te):
                overlaps.append((ps, pe))
        
        if overlaps:
            # Sort by start time
            overlaps.sort(key=lambda x: x[0])
            first_pred_start = overlaps[0][0]
            
            # Latency = Pred Start - True Start
            # If Pred Start < True Start (early detection?), latency is negative or 0.
            # Usually we care about delay, so max(0, ...)?
            # But let's just take the difference in time.
            
            t_true = timestamps.iloc[ts]
            t_pred = timestamps.iloc[first_pred_start]
            
            latency_seconds = (t_pred - t_true).total_seconds()
            latencies.append(latency_seconds / 60.0) # Minutes
            
    median_latency = np.median(latencies) if latencies else np.nan
    
    return {
        "fp_per_day": fp_per_day,
        "median_latency_minutes": median_latency,
        "fp_count": fp_count,
        "duration_days": duration_days
    }
