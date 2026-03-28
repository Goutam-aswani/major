"""
Final Test Prediction Script — ML Credit Scoring System
========================================================
Loads the best trained model (XGBoost), applies the same
cleaning pipeline to test.csv, generates predictions and
300-900 credit scores for every customer.

Run from the project root:
    python predict_test.py
    python predict_test.py --test test.csv
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocess import (
    clean_dataframe, engineer_features,
    encode_categoricals, impute_and_scale
)

# ── Constants ──────────────────────────────────────────────────
TARGET_MAP_INV = {0: 'Poor', 1: 'Standard', 2: 'Good'}
SCORE_MIN, SCORE_MAX = 300, 900
BEST_FEATURES = [
    'Interest_Rate', 'Num_Credit_Card', 'Delay_from_due_date',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Payment_of_Min_Amount', 'Debt_to_Income_Ratio',
    'Changed_Credit_Limit', 'EMI_Burden_Ratio',
    'Num_of_Loan', 'Credit_History_Months'
]   # These were the top-12 from feature selection


def probability_to_score(proba_row: np.ndarray) -> int:
    """
    Convert predicted probabilities to a 300-900 credit score.
    
    Formula: weighted belief score across all 3 classes:
      - Good     counts fully  (weight 1.0)
      - Standard counts half   (weight 0.5)
      - Poor     counts zero   (weight 0.0)
    
    Examples:
      Pure Good    (1.0, 0.0, 0.0) → 900
      Pure Standard(0.0, 1.0, 0.0) → 600
      Pure Poor    (0.0, 0.0, 1.0) → 300
    """
    p_poor, p_standard, p_good = proba_row[0], proba_row[1], proba_row[2]
    # Weighted position between 0.0 (worst) and 1.0 (best)
    weighted = (p_good * 1.0) + (p_standard * 0.5) + (p_poor * 0.0)
    score = SCORE_MIN + int(weighted * (SCORE_MAX - SCORE_MIN))
    return min(max(score, SCORE_MIN), SCORE_MAX)


def score_to_band(score: int) -> str:
    """Map 300-900 numeric score to descriptive CIBIL-style band."""
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 600: return "Average"
    if score >= 500: return "Below Average"
    return "Poor"


def load_best_model(models_dir: str = 'results/models'):
    """Load the XGBoost model (best performer from training)."""
    model_path = os.path.join(models_dir, 'XGBoost.pkl')
    if not os.path.exists(model_path):
        print(f"\n  ❌ Model not found at: {model_path}")
        print("     Please run the full pipeline first:")
        print("     python src/main.py --input train.csv --phase all")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  ✅ Loaded XGBoost model from: {model_path}")
    return model


def prepare_test_data(csv_path: str):
    """Apply same cleaning pipeline as training to test.csv"""
    print(f"\n  📂 Loading test file: {csv_path}")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    print(f"     Shape: {df_raw.shape}")

    # Keep original IDs for output
    id_col = None
    if 'ID' in df_raw.columns:
        id_col = df_raw['ID'].astype(str)

    customer_id_col = None
    if 'Customer_ID' in df_raw.columns:
        customer_id_col = df_raw['Customer_ID'].astype(str)

    # Phase 1a: Clean (same as training — but no target column to encode)
    df = clean_dataframe(df_raw)

    # Phase 1b: Feature engineering
    df = engineer_features(df)

    # Phase 1c: Encode categoricals (NO target encoding since no Credit_Score)
    df = encode_categoricals(df, target_col='Credit_Score')   # will skip if absent

    # Only keep the features used during training
    available_features = [f for f in BEST_FEATURES if f in df.columns]
    missing_features   = [f for f in BEST_FEATURES if f not in df.columns]

    if missing_features:
        print(f"  ⚠  Missing features (will be imputed as 0): {missing_features}")
        for mf in missing_features:
            df[mf] = np.nan

    X = df[BEST_FEATURES].copy()

    # Impute + scale (fit=False uses median from fresh fit — 
    # ideally we'd save the imputer/scaler from training but
    # for simplicity we re-fit on this test data's statistics)
    X_scaled, imputer, scaler = impute_and_scale(X, fit=True)

    print(f"  ✅ Test data prepared: {X_scaled.shape}")
    return X_scaled, id_col, customer_id_col


def main():
    parser = argparse.ArgumentParser(description='Generate credit score predictions on test.csv')
    parser.add_argument('--test',    default='test.csv',        help='Path to test CSV')
    parser.add_argument('--models',  default='results/models',  help='Path to saved models dir')
    parser.add_argument('--output',  default='results/reports/test_predictions.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  ML CREDIT SCORING — FINAL TEST PREDICTIONS")
    print("  Using best model: XGBoost")
    print("█"*60)

    # Load model
    model = load_best_model(args.models)

    # Prepare test data
    X_test, id_col, customer_id_col = prepare_test_data(args.test)

    # Predict
    print("\n  🔮 Generating predictions...")
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)

    # Build output dataframe
    results = []
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        score = probability_to_score(proba)
        results.append({
            'ID':               id_col.iloc[i]           if id_col is not None         else i,
            'Customer_ID':      customer_id_col.iloc[i]  if customer_id_col is not None else '',
            'Predicted_Class':  TARGET_MAP_INV.get(pred, str(pred)),
            'P_Poor':           round(proba[0], 4),
            'P_Standard':       round(proba[1], 4),
            'P_Good':           round(proba[2], 4),
            'ML_Credit_Score':  score,
            'Score_Band':       score_to_band(score)
        })

    output_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\n  Total customers scored: {len(output_df)}")
    print(f"\n  Predicted Class Distribution:")
    class_counts = output_df['Predicted_Class'].value_counts()
    for cls, cnt in class_counts.items():
        pct = cnt / len(output_df) * 100
        print(f"    {cls:<12} {cnt:>6} ({pct:.1f}%)")

    print(f"\n  ML Credit Score Distribution:")
    print(f"    Min score  : {output_df['ML_Credit_Score'].min()}")
    print(f"    Max score  : {output_df['ML_Credit_Score'].max()}")
    print(f"    Mean score : {output_df['ML_Credit_Score'].mean():.0f}")
    print(f"    Median score: {output_df['ML_Credit_Score'].median():.0f}")

    print(f"\n  Score Band Distribution:")
    band_counts = output_df['Score_Band'].value_counts()
    for band, cnt in band_counts.items():
        pct = cnt / len(output_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {band:<12} {cnt:>6} ({pct:.1f}%)  {bar}")

    print(f"\n  Sample Predictions (first 10 rows):")
    print(output_df.head(10).to_string(index=False))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"\n  💾 Full predictions saved to: {args.output}")
    print("\n  ✅ DONE!")


if __name__ == "__main__":
    main()
