import pickle
import pandas as pd
import numpy as np
import os
from src.data.preprocess import clean_dataframe, engineer_features, encode_categoricals, impute_and_scale

print("Running scaler export script...")

BEST_FEATURES = [
    'Interest_Rate', 'Num_Credit_Card', 'Delay_from_due_date',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Payment_of_Min_Amount', 'Debt_to_Income_Ratio',
    'Changed_Credit_Limit', 'EMI_Burden_Ratio',
    'Num_of_Loan', 'Credit_History_Months'
]

if not os.path.exists("test.csv"):
    print("test.csv not found!")
    exit(1)

df_raw = pd.read_csv("test.csv", low_memory=False)
df_clean = clean_dataframe(df_raw)
df_eng = engineer_features(df_clean)
df_enc = encode_categoricals(df_eng, target_col='Credit_Score')

for col in BEST_FEATURES:
    if col not in df_enc.columns:
        df_enc[col] = np.nan
        
X_ref = df_enc[BEST_FEATURES].copy()
_, imputer, scaler = impute_and_scale(X_ref, fit=True)

os.makedirs("results/models", exist_ok=True)
with open("results/models/imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)
with open("results/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
print("Successfully exported imputer.pkl and scaler.pkl to results/models/")
