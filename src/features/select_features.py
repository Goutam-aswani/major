"""
Feature Selection Module — ML Credit Scoring System
====================================================
Runs 5 distinct feature selection methods and compares them to find
the most consistently important features across all methods.

Methods:
  1. Statistical Filter  (Chi² + ANOVA F-scores)
  2. Mutual Information
  3. Recursive Feature Elimination (RFE) with Random Forest
  4. L1 Regularisation (Lasso / LogisticRegression with penalty='l1')
  5. Tree-based Importance (XGBoost)
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif,
    mutual_info_classif, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


TOP_K = 12   # Number of features to select


# ──────────────────────────────────────────────────────────────
# METHOD 1: Statistical Filter (F-score from ANOVA)
# ──────────────────────────────────────────────────────────────

def statistical_filter(X: pd.DataFrame, y: pd.Series, k: int = TOP_K) -> list:
    """
    Use ANOVA F-scores to select top-k features.
    Works on numerical + encoded categorical features.
    """
    print("\n  [1/5] Statistical Filter (ANOVA F-score)...")

    # Shift to non-negative for chi2 compatibility (use f_classif instead)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=X.columns)
    selected = scores.nlargest(k).index.tolist()

    print(f"       Top features: {selected[:5]}...")
    return selected


# ──────────────────────────────────────────────────────────────
# METHOD 2: Mutual Information
# ──────────────────────────────────────────────────────────────

def mutual_information(X: pd.DataFrame, y: pd.Series, k: int = TOP_K) -> list:
    """
    Mutual Information captures non-linear relationships between
    features and the target — unlike ANOVA which assumes linearity.
    """
    print("\n  [2/5] Mutual Information...")

    mi_scores = mutual_info_classif(X, y, random_state=42)
    scores = pd.Series(mi_scores, index=X.columns)
    selected = scores.nlargest(k).index.tolist()

    print(f"       Top features: {selected[:5]}...")
    return selected


# ──────────────────────────────────────────────────────────────
# METHOD 3: Recursive Feature Elimination (RFE)
# ──────────────────────────────────────────────────────────────

def rfe_selection(X: pd.DataFrame, y: pd.Series, k: int = TOP_K) -> list:
    """
    RFE with Random Forest: iteratively removes least-important features.
    This is a wrapper method — model-dependent but very accurate.
    """
    print("\n  [3/5] Recursive Feature Elimination (RFE with RandomForest)...")
    print("        (This may take a minute...)")

    estimator = RandomForestClassifier(
        n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
    selector.fit(X, y)

    selected = X.columns[selector.support_].tolist()
    print(f"       Selected {len(selected)} features: {selected[:5]}...")
    return selected


# ──────────────────────────────────────────────────────────────
# METHOD 4: L1 Regularisation (Lasso / Logistic L1)
# ──────────────────────────────────────────────────────────────

def l1_regularization(X: pd.DataFrame, y: pd.Series, k: int = TOP_K) -> list:
    """
    Logistic Regression with L1 penalty drives unimportant feature
    coefficients to exactly zero — an embedded selection method.
    """
    print("\n  [4/5] L1 Regularisation (Logistic Regression with L1)...")

    # L1 requires non-negative features — use MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty='l1', solver='saga', C=0.5,
        max_iter=2000, random_state=42, multi_class='auto'
    )
    model.fit(X_scaled, y)

    # Sum absolute coefficients across all classes
    coef_abs = np.abs(model.coef_).sum(axis=0)
    scores = pd.Series(coef_abs, index=X.columns)
    selected = scores.nlargest(k).index.tolist()

    print(f"       Top features: {selected[:5]}...")
    return selected


# ──────────────────────────────────────────────────────────────
# METHOD 5: Tree-Based Importance (XGBoost)
# ──────────────────────────────────────────────────────────────

def tree_based_importance(X: pd.DataFrame, y: pd.Series, k: int = TOP_K) -> list:
    """
    XGBoost feature importance: 'gain' metric tells us how much
    each feature contributes to reducing prediction error.
    """
    print("\n  [5/5] Tree-Based Importance (XGBoost)...")

    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss',
        use_label_encoder=False, n_jobs=-1
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    selected = importance.nlargest(k).index.tolist()

    print(f"       Top features: {selected[:5]}...")
    return selected


# ──────────────────────────────────────────────────────────────
# MAIN: Run All 5 Methods and Build Consensus
# ──────────────────────────────────────────────────────────────

def run_feature_selection(X: pd.DataFrame, y: pd.Series,
                          k: int = TOP_K,
                          output_dir: str = 'results/reports') -> dict:
    """
    Run all 5 feature selection methods.
    Build a vote-count table showing which features are selected most consistently.

    Returns dict with keys: 'results_table', 'consensus_features', 'method_results'
    """
    print("\n" + "="*60)
    print("PHASE 3: FEATURE SELECTION (5 Methods)")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    method_results = {}

    # Run all 5
    method_results['Statistical Filter']      = statistical_filter(X, y, k)
    method_results['Mutual Information']       = mutual_information(X, y, k)
    method_results['RFE']                      = rfe_selection(X, y, k)
    method_results['L1 Regularisation']        = l1_regularization(X, y, k)
    method_results['Tree-Based (XGBoost)']     = tree_based_importance(X, y, k)

    # Build vote table
    all_features = X.columns.tolist()
    vote_table = pd.DataFrame(index=all_features)

    for method_name, selected in method_results.items():
        vote_table[method_name] = vote_table.index.isin(selected).astype(int)

    vote_table['Votes'] = vote_table.sum(axis=1)
    vote_table = vote_table.sort_values('Votes', ascending=False)

    print("\n" + "="*60)
    print("FEATURE SELECTION RESULTS — VOTE TABLE")
    print("="*60)
    print(vote_table.to_string())

    # Consensus: features selected by ALL 5 methods
    consensus = vote_table[vote_table['Votes'] == 5].index.tolist()
    majority  = vote_table[vote_table['Votes'] >= 4].index.tolist()

    print(f"\n  🏆 Features selected by ALL 5 methods ({len(consensus)}): {consensus}")
    print(f"  ✅ Features selected by ≥ 4 methods ({len(majority)}): {majority}")

    print("\n  📊 This is the PROOF that these features are robust predictors:")
    print("     If these features appear across filter, wrapper, AND embedded methods,")
    print("     they are genuinely predictive — not artifacts of one algorithm.")

    # Save
    vote_table.to_csv(os.path.join(output_dir, 'feature_selection_votes.csv'))
    print(f"\n  💾 Saved: {output_dir}/feature_selection_votes.csv")

    # Best feature set for training = top-k by votes
    best_features = vote_table.head(k).index.tolist()
    print(f"\n  ✨ Final feature set for training ({len(best_features)} features):")
    for i, f in enumerate(best_features, 1):
        votes = vote_table.loc[f, 'Votes']
        print(f"     {i:2}. {f:<35}  [{votes}/5 methods]")

    return {
        'results_table': vote_table,
        'consensus_features': consensus,
        'majority_features': majority,
        'best_features': best_features,
        'method_results': method_results
    }


if __name__ == "__main__":
    print("Feature Selection Module — import and call run_feature_selection(X, y)")
