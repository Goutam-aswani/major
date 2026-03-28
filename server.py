import os
import sys
import pickle
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="ML Credit Scorer")

BEST_FEATURES = [
    'Interest_Rate', 'Num_Credit_Card', 'Delay_from_due_date',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Payment_of_Min_Amount', 'Debt_to_Income_Ratio',
    'Changed_Credit_Limit', 'EMI_Burden_Ratio',
    'Num_of_Loan', 'Credit_History_Months'
]

print("Loading model and JSON scaler weights...")
try:
    with open("results/models/XGBoost.pkl", "rb") as f:
        model = pickle.load(f)
    with open("results/models/scaler_weights.json", "r") as f:
        weights = json.load(f)
        
    imputer_stats = weights["imputer_statistics"]
    scaler_center = weights["scaler_center"]
    scaler_scale = weights["scaler_scale"]
    
    print("Startup sequence complete.")
except Exception as e:
    print(f"Error loading model or weights: {e}")
    sys.exit(1)

class CustomerData(BaseModel):
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Total_EMI_per_month: float
    Interest_Rate: float
    Num_Credit_Card: float
    Delay_from_due_date: float
    Num_Credit_Inquiries: float
    Credit_Mix: str
    Outstanding_Debt: float
    Payment_of_Min_Amount: str
    Changed_Credit_Limit: float
    Num_of_Loan: float
    Credit_History_Months: float

def compute_ml_score(proba):
    p_poor, p_standard, p_good = proba[0], proba[1], proba[2]
    weighted = (p_good * 1.0) + (p_standard * 0.5) + (p_poor * 0.0)
    score = 300 + int(weighted * 600)
    return min(max(score, 300), 900)

def score_to_band(score: int) -> str:
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 600: return "Average"
    if score >= 500: return "Below Average"
    return "Poor"

@app.post("/api/predict")
async def predict_score(data: CustomerData):
    try:
        mix_map = {"Bad": 0, "Standard": 1, "Good": 2}
        min_map = {"Yes": 1, "No": 0, "Not Meaningful": 0.5}

        encoded_mix = mix_map.get(data.Credit_Mix, 1)
        encoded_min = min_map.get(data.Payment_of_Min_Amount, 0.5)

        dti = data.Outstanding_Debt / data.Annual_Income if data.Annual_Income > 0 else 0
        emi = data.Total_EMI_per_month / data.Monthly_Inhand_Salary if data.Monthly_Inhand_Salary > 0 else 0

        # Construct pure python array matching exact ordered features
        raw_values = [
            data.Interest_Rate,
            data.Num_Credit_Card,
            data.Delay_from_due_date,
            data.Num_Credit_Inquiries,
            encoded_mix,
            data.Outstanding_Debt,
            encoded_min,
            dti,
            data.Changed_Credit_Limit,
            emi,
            data.Num_of_Loan,
            data.Credit_History_Months
        ]

        # Manual Impute & Scale (Saves 350MB of Scikit-Learn + Scipy + Pandas)
        scaled_features = []
        for i in range(len(raw_values)):
            val = raw_values[i]
            
            # 1. Impute NaNs with median statistic
            if val is None or np.isnan(val):
                val = imputer_stats[i]
                
            # 2. Scale via RobustScaler formula (X - center) / scale
            scaled_val = (val - scaler_center[i]) / scaler_scale[i] if scaler_scale[i] != 0 else val
            scaled_features.append(scaled_val)
            
        # Predict directly using numpy 2D array
        X_in = np.array([scaled_features])
        probas = model.predict_proba(X_in)[0]
        
        score = compute_ml_score(probas)
        band = score_to_band(score)

        return {
            "score": score,
            "band": band,
            "metrics": {
                "dti": float(round(dti * 100, 2)),
                "emi": float(round(emi * 100, 2)),
                "credit_age_years": float(round(data.Credit_History_Months / 12, 1))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Static serving for root assets
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/style.css")
def serve_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
def serve_js():
    return FileResponse("script.js", media_type="application/javascript")

@app.get("/")
def serve_index():
    return FileResponse("index.html")
