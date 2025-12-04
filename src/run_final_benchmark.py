import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_gp import run_gp_pipeline
from src.run_lstm import run_lstm_pipeline
from src.run_hybrid import run_hybrid_ensemble
from src.run_bsts import run_bsts_pipeline
from src.aggregate_metrics import aggregate_metrics

def run_benchmark():
    # Robustly determine project root and NAB path
    current_dir = Path(os.getcwd())
    if current_dir.name == 'src':
        project_root = current_dir.parent
    else:
        project_root = current_dir

    nab_root = str(project_root / 'NAB')
    
    datasets = [
        {
            "name": "NYC Taxi",
            "file_key": "realKnownCause/nyc_taxi.csv",
            "seasonal_periods": [48, 336] # Daily + Weekly
        },
        {
            "name": "Machine Temp",
            "file_key": "realKnownCause/machine_temperature_system_failure.csv",
            "seasonal_periods": [48] # Daily only (likely)
        },
        {
            "name": "Twitter Volume",
            "file_key": "realTweets/Twitter_volume_AMZN.csv",
            "seasonal_periods": [48, 336] # Daily + Weekly
        }
    ]
    
    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"Running Benchmark for: {ds['name']}")
        print(f"{'='*50}\n")
        
        file_key = ds['file_key']
        seasonal = ds['seasonal_periods']
        
        # 1. GP
        print(f"\n--- Running GP for {ds['name']} ---")
        try:
            run_gp_pipeline(nab_root, file_key, save_dir="./results/gp")
        except Exception as e:
            print(f"GP Failed: {e}")
            
        # 2. LSTM
        print(f"\n--- Running LSTM for {ds['name']} ---")
        try:
            run_lstm_pipeline(nab_root, file_key, save_dir="./results/lstm")
        except Exception as e:
            print(f"LSTM Failed: {e}")
            
        # 3. Hybrid
        print(f"\n--- Running Hybrid for {ds['name']} ---")
        try:
            run_hybrid_ensemble(nab_root, file_key, save_dir="./results/hybrid")
        except Exception as e:
            print(f"Hybrid Failed: {e}")
            
        # 4. BSTS
        print(f"\n--- Running BSTS for {ds['name']} ---")
        try:
            run_bsts_pipeline(nab_root, file_key, save_dir="./results/bsts", seasonal_periods=seasonal)
        except Exception as e:
            print(f"BSTS Failed: {e}")
            
    print("\n{'='*50}")
    print("Benchmark Complete. Aggregating Metrics...")
    print(f"{'='*50}\n")
    
    aggregate_metrics()

if __name__ == "__main__":
    run_benchmark()
