"""
Model Training Module — ML Credit Scoring System
=================================================
Trains all 7 ML algorithms with 5-fold cross-validation.
Also trains the CIBIL Baseline model for comparison.

Models:
  1. Logistic Regression  (simple baseline — closest to CIBIL formula)
  2. Decision Tree        (fully explainable rule tree)
  3. Random Forest        (ensemble, reduces overfitting)
  4. XGBoost              (best-in-class for tabular data)
  5. SVM                  (non-linear kernel machine)
  6. KNN                  (instance-based similarity)
  7. Neural Network MLP   (deep pattern recognition)
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return all 7 models with pre-tuned hyperparameters."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, multi_class='auto'),

        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_split=50, random_state=42),

        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=30,
            random_state=42, n_jobs=-1),

        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, n_jobs=-1),

        'SVM': SVC(
            kernel='rbf', C=1.0, probability=True, random_state=42),

        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=-1),

        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            max_iter=500, random_state=42, early_stopping=True)
    }


def get_cibil_baseline_model() -> LogisticRegression:
    """Returns a simple logistic regression to simulate CIBIL's fixed formula."""
    return LogisticRegression(max_iter=1000, C=0.1, random_state=42)


CIBIL_FEATURES = [
    'Num_of_Loan', 'Delay_from_due_date',
    'Outstanding_Debt', 'Num_of_Delayed_Payment'
]


# ──────────────────────────────────────────────────────────────
# Cross-validation scoring
# ──────────────────────────────────────────────────────────────

def evaluate_with_cv(model, X: pd.DataFrame, y: pd.Series,
                     n_splits: int = 5) -> dict:
    """
    Run stratified k-fold CV and return averaged metrics.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        'accuracy':  'accuracy',
        'f1_macro':  'f1_macro',
        'precision': 'precision_macro',
        'recall':    'recall_macro',
        'roc_auc':   'roc_auc_ovr_weighted'
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring,
                            return_train_score=False, n_jobs=-1)
    return {
        'CV_Accuracy':  round(scores['test_accuracy'].mean(), 4),
        'CV_F1_Macro':  round(scores['test_f1_macro'].mean(), 4),
        'CV_Precision': round(scores['test_precision'].mean(), 4),
        'CV_Recall':    round(scores['test_recall'].mean(), 4),
        'CV_ROC_AUC':   round(scores['test_roc_auc'].mean(), 4),
        'CV_Std_F1':    round(scores['test_f1_macro'].std(), 4)
    }


# ──────────────────────────────────────────────────────────────
# Holdout evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_on_holdout(model, X_test: pd.DataFrame, y_test: pd.Series,
                        target_names: list = None) -> dict:
    """Evaluate a fitted model on the holdout test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    target_names = target_names or ['Poor', 'Standard', 'Good']

    metrics = {
        'Test_Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Test_F1_Macro':  round(f1_score(y_test, y_pred, average='macro'), 4),
        'Test_Precision': round(precision_score(y_test, y_pred, average='macro'), 4),
        'Test_Recall':    round(recall_score(y_test, y_pred, average='macro'), 4),
    }

    if y_prob is not None:
        metrics['Test_ROC_AUC'] = round(
            roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'), 4)

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    # Add per-class F1 for Poor (class 0) — most important for credit risk
    metrics['Test_F1_Poor']     = round(report['Poor']['f1-score'], 4)
    metrics['Test_F1_Standard'] = round(report['Standard']['f1-score'], 4)
    metrics['Test_F1_Good']     = round(report['Good']['f1-score'], 4)

    return metrics, report


# ──────────────────────────────────────────────────────────────
# MAIN: Train all models + CIBIL baseline
# ──────────────────────────────────────────────────────────────

def run_training(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test:  pd.DataFrame, y_test:  pd.Series,
                 best_features: list,
                 output_dir: str = 'results/models',
                 reports_dir: str = 'results/reports') -> dict:
    """
    Train all 7 models and the CIBIL baseline.
    Evaluate with 5-fold CV and on holdout set.
    Save models and return full comparison table.

    Parameters
    ----------
    X_train, y_train : training data
    X_test,  y_test  : holdout test data
    best_features    : top feature list from feature selection
    output_dir       : where to save .pkl model files
    reports_dir      : where to save comparison CSV

    Returns
    -------
    dict with keys: 'comparison_table', 'fitted_models', 'best_model_name'
    """
    print("\n" + "="*60)
    print("PHASE 4: MODEL TRAINING — 7 Algorithms + CIBIL Baseline")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    models = get_models()
    all_results = []
    fitted_models = {}

    # ── A. Train CIBIL Baseline first ─────────────────────────
    print("\n  ⚙  CIBIL Baseline (Traditional Features Only):")
    cibil_cols = [c for c in CIBIL_FEATURES if c in X_train.columns]
    cibil_model = get_cibil_baseline_model()

    t0 = time.time()
    cv_metrics_cibil = evaluate_with_cv(cibil_model, X_train[cibil_cols], y_train)
    cibil_model.fit(X_train[cibil_cols], y_train)
    holdout_metrics_cibil, _ = evaluate_on_holdout(cibil_model, X_test[cibil_cols], y_test)
    elapsed = time.time() - t0

    row_cibil = {'Model': '⚖ CIBIL Baseline (Rule-Based)'}
    row_cibil.update(cv_metrics_cibil)
    row_cibil.update(holdout_metrics_cibil)
    row_cibil['Training_Time_s'] = round(elapsed, 2)
    all_results.append(row_cibil)
    fitted_models['CIBIL Baseline'] = cibil_model

    print(f"     CV F1 (macro): {cv_metrics_cibil['CV_F1_Macro']:.4f}")
    print(f"     Test Accuracy: {holdout_metrics_cibil['Test_Accuracy']:.4f}")

    # ── B. Train all 7 ML models ──────────────────────────────
    # Use best_features subset selected in Phase 3
    X_train_sel = X_train[best_features]
    X_test_sel  = X_test[best_features]

    for model_name, model in models.items():
        print(f"\n  ⚙  {model_name}...")
        t0 = time.time()

        # 5-fold CV
        cv_metrics = evaluate_with_cv(model, X_train_sel, y_train)
        # Fit on full training set
        model.fit(X_train_sel, y_train)
        # Holdout
        holdout_metrics, report = evaluate_on_holdout(model, X_test_sel, y_test)
        elapsed = time.time() - t0

        row = {'Model': model_name}
        row.update(cv_metrics)
        row.update(holdout_metrics)
        row['Training_Time_s'] = round(elapsed, 2)
        all_results.append(row)
        fitted_models[model_name] = model

        # Save model
        model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"     CV F1: {cv_metrics['CV_F1_Macro']:.4f} ± {cv_metrics['CV_Std_F1']:.4f}")
        print(f"     Test Accuracy: {holdout_metrics['Test_Accuracy']:.4f}  |  "
              f"F1-Poor: {holdout_metrics['Test_F1_Poor']:.4f}")

    # ── C. Build comparison table ──────────────────────────────
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('Test_F1_Macro', ascending=False)

    print("\n" + "="*60)
    print("PHASE 4 RESULTS: MODEL COMPARISON TABLE")
    print("="*60)
    display_cols = ['Model', 'CV_F1_Macro', 'Test_Accuracy', 'Test_F1_Macro',
                    'Test_F1_Poor', 'Test_ROC_AUC', 'Training_Time_s']
    display_cols = [c for c in display_cols if c in comparison_df.columns]
    print(comparison_df[display_cols].to_string(index=False))

    # Save comparison
    comparison_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)
    print(f"\n  💾 Saved: {reports_dir}/model_comparison.csv")

    # Best model (excluding baseline)
    ml_models_df = comparison_df[~comparison_df['Model'].str.contains('CIBIL')]
    best_model_name = ml_models_df.iloc[0]['Model']
    cibil_f1 =  comparison_df[comparison_df['Model'].str.contains('CIBIL')]['Test_F1_Macro'].values[0]
    best_f1   = ml_models_df.iloc[0]['Test_F1_Macro']
    improvement = ((best_f1 - cibil_f1) / cibil_f1) * 100

    print(f"\n  🏆 Best Model: {best_model_name}  (Test F1={best_f1:.4f})")
    print(f"  ⚖  CIBIL Baseline F1: {cibil_f1:.4f}")
    print(f"  📈 Improvement over CIBIL: +{improvement:.1f}%")

    return {
        'comparison_table': comparison_df,
        'fitted_models': fitted_models,
        'best_model_name': best_model_name,
        'best_features': best_features,
        'cibil_features': cibil_cols
    }


if __name__ == "__main__":
    print("Model Training Module — import and call run_training()")
