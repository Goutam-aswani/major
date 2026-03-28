import os
import pickle
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.models.interpret import run_interpretability

BEST_FEATURES = [
    'Interest_Rate', 'Num_Credit_Card', 'Delay_from_due_date',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Payment_of_Min_Amount', 'Debt_to_Income_Ratio',
    'Changed_Credit_Limit', 'EMI_Burden_Ratio',
    'Num_of_Loan', 'Credit_History_Months'
]

def main():
    print("Loading data...")
    X = pd.read_csv('data/processed/X_processed.csv')
    y = pd.read_csv('data/processed/y_processed.csv').squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    print("Loading XGBoost model...")
    with open('results/models/XGBoost.pkl', 'rb') as f:
        model = pickle.load(f)

    # Note: probability_to_score and score_to_band in src.models.interpret might have the old formula.
    # Let me monkey patch them runtime to use the correct formula as in predict_test.py
    import src.models.interpret as interp
    def new_prob_to_score(proba_row):
        p_poor, p_standard, p_good = proba_row[0], proba_row[1], proba_row[2]
        weighted = (p_good * 1.0) + (p_standard * 0.5) + (p_poor * 0.0)
        score = 300 + int(weighted * 600)
        return min(max(score, 300), 900)
        
    def new_score_to_band(score):
        if score >= 750: return "Excellent"
        if score >= 700: return "Good"
        if score >= 650: return "Fair"
        if score >= 600: return "Average"
        if score >= 500: return "Below Average"
        return "Poor"
        
    interp.probability_to_score = new_prob_to_score
    interp.score_to_band = new_score_to_band

    print("Running interpretability...")
    run_interpretability(
        model, 
        X_train[BEST_FEATURES], 
        X_test[BEST_FEATURES], 
        y_test, 
        BEST_FEATURES, 
        target_map_inv={0: 'Poor', 1: 'Standard', 2: 'Good'},
        figures_dir='results/figures',
        reports_dir='results/reports'
    )

if __name__ == '__main__':
    main()
