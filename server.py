import os
import sys
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


# Define FastAPI app
app = FastAPI(title="ML Credit Scorer")

# Ordered feature list exact as expected by XGBoost
BEST_FEATURES = [
    'Interest_Rate', 'Num_Credit_Card', 'Delay_from_due_date',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Payment_of_Min_Amount', 'Debt_to_Income_Ratio',
    'Changed_Credit_Limit', 'EMI_Burden_Ratio',
    'Num_of_Loan', 'Credit_History_Months'
]

# Startup logic: Load model and construct Scaler
print("Loading model and constructing Scaler...")
try:
    with open("results/models/XGBoost.pkl", "rb") as f:
        model = pickle.load(f)
    with open("results/models/imputer.pkl", "rb") as f:
        imputer = pickle.load(f)
    with open("results/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    print("Startup sequence complete. Scaler and Model ready.")
    
except Exception as e:
    print(f"Error loading model or scaler: {e}")
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
        # Categorical Encodings
        mix_map = {"Bad": 0, "Standard": 1, "Good": 2}
        min_map = {"Yes": 1, "No": 0, "Not Meaningful": 0.5}

        encoded_mix = mix_map.get(data.Credit_Mix, 1)
        encoded_min = min_map.get(data.Payment_of_Min_Amount, 0.5)

        # Engineered features
        dti = data.Outstanding_Debt / data.Annual_Income if data.Annual_Income > 0 else 0
        emi = data.Total_EMI_per_month / data.Monthly_Inhand_Salary if data.Monthly_Inhand_Salary > 0 else 0

        # Construct DataFrame directly in the exact feature order
        feature_dict = {
            'Interest_Rate': [data.Interest_Rate],
            'Num_Credit_Card': [data.Num_Credit_Card],
            'Delay_from_due_date': [data.Delay_from_due_date],
            'Num_Credit_Inquiries': [data.Num_Credit_Inquiries],
            'Credit_Mix': [encoded_mix],
            'Outstanding_Debt': [data.Outstanding_Debt],
            'Payment_of_Min_Amount': [encoded_min],
            'Debt_to_Income_Ratio': [dti],
            'Changed_Credit_Limit': [data.Changed_Credit_Limit],
            'EMI_Burden_Ratio': [emi],
            'Num_of_Loan': [data.Num_of_Loan],
            'Credit_History_Months': [data.Credit_History_Months]
        }
        
        df_in = pd.DataFrame(feature_dict)[BEST_FEATURES]
        
        # KEY FIX: Scale the incoming API data exactly how the model was trained!
        df_in_imputed = imputer.transform(df_in)
        df_in_scaled = scaler.transform(df_in_imputed)
        
        # Predict
        probas = model.predict_proba(df_in_scaled)[0]
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

# Mount static frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    print("Starting ML Score Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
