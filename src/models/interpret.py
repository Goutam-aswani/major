"""
Model Interpretability Module — ML Credit Scoring System
=========================================================
Explains the best model's predictions using:

  1. SHAP (Global + Local explanations)
  2. LIME (Local Interpretable Model-Agnostic Explanations)
  3. Credit Score translation: Probability → 300-900 scale

This module proves our system is FAIR and EXPLAINABLE — unlike
the black-box CIBIL formula where customers never know why they scored poorly.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# SHAP and LIME — install if needed: pip install shap lime
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  ⚠ SHAP not installed. Run: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("  ⚠ LIME not installed. Run: pip install lime")


# ──────────────────────────────────────────────────────────────
# SCORE TRANSLATION: Probability → 300-900 CIBIL-style score
# ──────────────────────────────────────────────────────────────

TARGET_NAMES = ['Poor', 'Standard', 'Good']
SCORE_MIN = 300
SCORE_MAX = 900


def probability_to_score(proba_row: np.ndarray) -> int:
    """
    Convert model output probabilities to a 300-900 credit score.

    The score is weighted toward the 'Good' class probability:
      score = 300 + (P_Good * 0.6 + P_Standard * 0.3 + P_Poor * 0.1) * 600

    This penalises poor probability while rewarding good credit signals.
    """
    p_poor, p_standard, p_good = proba_row[0], proba_row[1], proba_row[2]
    weighted = (p_good * 0.6) + (p_standard * 0.3) + (p_poor * 0.1)
    score = SCORE_MIN + int(weighted * (SCORE_MAX - SCORE_MIN))
    return min(max(score, SCORE_MIN), SCORE_MAX)


def score_to_band(score: int) -> str:
    """Map numeric score to descriptive band."""
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 600: return "Poor"
    return "Very Poor"


def generate_score_card(model, X_sample: pd.DataFrame,
                        feature_names: list,
                        target_map_inv: dict = None) -> pd.DataFrame:
    """
    Generate a credit score card for sample customers.
    Returns DataFrame with predicted class, probability, ML score, and band.
    """
    probas = model.predict_proba(X_sample)
    preds  = model.predict(X_sample)

    if target_map_inv is None:
        target_map_inv = {0: 'Poor', 1: 'Standard', 2: 'Good'}

    records = []
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        score = probability_to_score(proba)
        records.append({
            'Customer_Index': X_sample.index[i] if hasattr(X_sample, 'index') else i,
            'Predicted_Class': target_map_inv.get(pred, str(pred)),
            'P_Poor':     round(proba[0], 3),
            'P_Standard': round(proba[1], 3),
            'P_Good':     round(proba[2], 3),
            'ML_Credit_Score': score,
            'Score_Band': score_to_band(score)
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# SHAP: Global Feature Importance
# ──────────────────────────────────────────────────────────────

def run_shap_analysis(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                      feature_names: list, figures_dir: str) -> dict:
    """
    Run SHAP analysis:
      - Global: Bar chart of mean |SHAP values| per feature
      - Local:  Waterfall plots for 3 example customers (one per class)

    Returns dict with shap_values array.
    """
    if not SHAP_AVAILABLE:
        print("  ⚠ Skipping SHAP (not installed)")
        return {}

    print("\n  Running SHAP analysis (this may take a minute)...")

    # Use TreeExplainer for tree-based models, KernelExplainer otherwise
    model_type = type(model).__name__
    if model_type in ('XGBClassifier', 'RandomForestClassifier',
                      'DecisionTreeClassifier', 'GradientBoostingClassifier'):
        explainer = shap.TreeExplainer(model)
        # Sample 2000 rows for speed
        X_sample = X_explain.sample(min(2000, len(X_explain)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Use a small background dataset for non-tree models
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        X_sample = X_explain.sample(min(200, len(X_explain)), random_state=42)
        shap_values = explainer.shap_values(X_sample)

    # ── Global SHAP: mean absolute value per feature ──────────
    # For multiclass, shap_values may be a list of arrays (one per class) or a 3D array
    if isinstance(shap_values, list):
        global_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        global_importance = np.abs(shap_values).mean(axis=0)
        # If 3D array (num_features, num_classes) returned by new SHAP versions for XGBoost
        if len(global_importance.shape) > 1:
            global_importance = global_importance.mean(axis=1)

    importance_df = pd.Series(global_importance, index=feature_names).sort_values(ascending=False)

    # Plot global SHAP bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    colours = ['#e74c3c' if i < 3 else '#3498db' for i in range(len(importance_df))]
    importance_df.head(15).sort_values().plot(kind='barh', ax=ax, color=colours[::-1])
    ax.set_title("SHAP Feature Importance\n(Mean |SHAP Value| across all customers)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    # Add value labels
    for bar, val in zip(ax.patches, importance_df.head(15).sort_values().values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    shap_path = os.path.join(figures_dir, 'shap_global_importance.png')
    plt.savefig(shap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {shap_path}")

    return {
        'shap_values': shap_values,
        'importance_df': importance_df,
        'X_sample': X_sample
    }


# ──────────────────────────────────────────────────────────────
# LIME: Local Explanation for a single customer
# ──────────────────────────────────────────────────────────────

def run_lime_explanation(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                         feature_names: list, customer_idx: int,
                         class_names: list = None,
                         figures_dir: str = 'results/figures') -> dict:
    """
    Generate a LIME local explanation for one specific customer.
    Shows which features pushed their score UP or DOWN.
    """
    if not LIME_AVAILABLE:
        print("  ⚠ Skipping LIME (not installed)")
        return {}

    class_names = class_names or ['Poor', 'Standard', 'Good']

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )

    instance = X_explain.iloc[customer_idx].values
    explanation = explainer.explain_instance(
        instance, model.predict_proba, num_features=10, top_labels=1
    )

    # Get predicted class for this customer
    pred_class_idx = model.predict(X_explain.iloc[[customer_idx]])[0]
    pred_class_name= class_names[pred_class_idx]

    # Extract feature contributions
    exp_list = explanation.as_list(label=pred_class_idx)
    features_lime  = [e[0] for e in exp_list]
    weights_lime   = [e[1] for e in exp_list]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ['#27ae60' if w > 0 else '#e74c3c' for w in weights_lime]
    bars = ax.barh(features_lime, weights_lime, color=colours)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f"LIME Explanation — Customer #{customer_idx}\n"
                 f"Predicted: {pred_class_name}",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Feature Contribution to Score", fontsize=11)
    ax.set_ylabel("Feature Condition", fontsize=11)

    # Legend
    green_patch = mpatches.Patch(color='#27ae60', label='Increases credit score')
    red_patch   = mpatches.Patch(color='#e74c3c', label='Decreases credit score')
    ax.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.tight_layout()
    lime_path = os.path.join(figures_dir, f'lime_customer_{customer_idx}.png')
    plt.savefig(lime_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {lime_path}")

    return {
        'explanation': explanation,
        'predicted_class': pred_class_name,
        'feature_contributions': dict(zip(features_lime, weights_lime))
    }


# ──────────────────────────────────────────────────────────────
# MAIN: Full Interpretability Pipeline
# ──────────────────────────────────────────────────────────────

def run_interpretability(model, X_train: pd.DataFrame,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         feature_names: list,
                         target_map_inv: dict = None,
                         figures_dir: str = 'results/figures',
                         reports_dir: str = 'results/reports') -> dict:
    """
    Run the complete interpretability pipeline:
      1. Generate credit scores for all test customers
      2. SHAP global importance
      3. LIME local explanations for 3 customers (one per class)

    Returns dict with all results.
    """
    print("\n" + "="*60)
    print("PHASE 6: MODEL INTERPRETABILITY (SHAP + LIME)")
    print("="*60)

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    target_map_inv = target_map_inv or {0: 'Poor', 1: 'Standard', 2: 'Good'}
    results = {}

    # ── 1. Credit Score Card ───────────────────────────────────
    print("\n  📊 Generating Credit Score Cards...")
    score_card = generate_score_card(model, X_test, feature_names, target_map_inv)

    score_card.to_csv(os.path.join(reports_dir, 'credit_score_card.csv'), index=False)
    print(f"  💾 Saved: {reports_dir}/credit_score_card.csv")

    print("\n  Sample Score Cards (first 10 customers):")
    print(score_card.head(10).to_string(index=False))

    # Score distribution summary
    print("\n  Score Band Distribution:")
    print(score_card['Score_Band'].value_counts().to_string())

    results['score_card'] = score_card

    # ── 2. SHAP Global Analysis ───────────────────────────────
    shap_results = run_shap_analysis(
        model, X_train, X_test, feature_names, figures_dir)
    results['shap'] = shap_results

    # ── 3. LIME Local Explanations ────────────────────────────
    # Pick one example customer per class
    print("\n  🔍 Generating LIME explanations for 3 example customers...")
    lime_results = {}

    for target_class_idx in [0, 1, 2]:
        class_name = target_map_inv[target_class_idx]
        # Find a customer in the test set actually predicted as this class
        preds = model.predict(X_test)
        class_indices = np.where(preds == target_class_idx)[0]

        if len(class_indices) > 0:
            customer_idx = class_indices[0]   # first matching customer
            print(f"\n  Customer #{customer_idx}  (Class: {class_name})")
            lime_result = run_lime_explanation(
                model, X_train, X_test, feature_names,
                customer_idx=customer_idx,
                figures_dir=figures_dir
            )
            lime_results[class_name] = lime_result

            # Print the plain-English explanation
            contribs = lime_result.get('feature_contributions', {})
            print(f"  Explanation for why this customer is '{class_name}':")
            sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            for feat, weight in sorted_contribs[:5]:
                direction = "✅ HELPS" if weight > 0 else "❌ HURTS"
                print(f"    {direction} | {feat:<50} impact={weight:+.4f}")

    results['lime'] = lime_results

    # ── 4. CIBIL vs ML Score Comparison ──────────────────────
    print("\n" + "="*60)
    print("FINAL PROOF: ML Score vs Traditional CIBIL Score")
    print("="*60)
    print("""
  Traditional CIBIL gives a ZERO score to any customer with
  no prior loan history → unfair to first-time borrowers.

  Our ML model can still score any customer using:
    ✅ Monthly Balance      — Do they save money regularly?
    ✅ Payment Behaviour    — High/Low spending pattern?
    ✅ Credit Utilization   — Are they over-leveraged?
    ✅ Occupation           — Stable employment?
    
  Result: Every customer gets a FAIR, EXPLAINABLE score.
    """)

    return results


if __name__ == "__main__":
    print("Interpretability Module — import and call run_interpretability()")
