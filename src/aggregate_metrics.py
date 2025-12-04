import json
import pandas as pd
from pathlib import Path
import os

def aggregate_metrics(results_dir="./results"):
    results_path = Path(results_dir)
    metrics_data = []
    
    print(f"Scanning {results_path} for metrics.json...")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(results_path):
        if "metrics.json" in files:
            file_path = Path(root) / "metrics.json"
            
            # Infer model name from directory structure
            # e.g. ./results/bsts/realKnownCause__nyc_taxi.csv/metrics.json
            # parent = realKnownCause__nyc_taxi.csv
            # parent.parent = bsts (Model Name)
            
            model_name = Path(root).parent.name
            dataset_name = Path(root).name
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                row = {
                    "Model": model_name.upper(),
                    "Dataset": dataset_name
                }
                
                # Event Level Metrics
                if "event_level" in data:
                    evt = data["event_level"]
                    row["Event_F1"] = evt.get("f1", 0)
                    row["Precision"] = evt.get("precision", 0)
                    row["Recall"] = evt.get("recall", 0)
                    
                # Business Metrics
                if "business" in data:
                    bus = data["business"]
                    row["FP_per_Day"] = bus.get("fp_per_day", 0)
                    row["Latency_Min"] = bus.get("median_latency_minutes", 0)
                elif "fp_per_day" in data: # Handle flat structure if any
                    row["FP_per_Day"] = data.get("fp_per_day", 0)
                    row["Latency_Min"] = data.get("median_latency_minutes", 0)
                    
                metrics_data.append(row)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    if not metrics_data:
        print("No metrics found.")
        return
        
    df = pd.DataFrame(metrics_data)
    
    # Sort by F1
    if "Event_F1" in df.columns:
        df = df.sort_values("Event_F1", ascending=False)
        
    # Save
    out_path = results_path / "metrics_summary.csv"
    df.to_csv(out_path, index=False)
    
    print("\n--- Final Metrics Summary ---")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    aggregate_metrics()
