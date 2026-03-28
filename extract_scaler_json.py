import pickle
import json
import numpy as np

print("Extracting Scaler Weights to JSON...")

try:
    with open("results/models/imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
        
    with open("results/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    # Extact values, handling potential numpy types (float32/64)
    # SimpleImputer with median strategy uses .statistics_
    statistics = [float(x) if not np.isnan(x) else None for x in imputer.statistics_]
    
    # RobustScaler uses .center_ and .scale_
    center = [float(x) for x in scaler.center_]
    scale = [float(x) for x in scaler.scale_]
    
    weights = {
        "imputer_statistics": statistics,
        "scaler_center": center,
        "scaler_scale": scale
    }
    
    with open("results/models/scaler_weights.json", "w") as f:
        json.dump(weights, f, indent=4)
        
    print("Successfully exported scaler_weights.json")
    
except Exception as e:
    print(f"Error: {e}")
