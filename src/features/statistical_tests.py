"""
Statistical Testing Module — ML Credit Scoring System
======================================================
Implements all three statistical tests to PROVE which features
drive credit quality and justify ML over a fixed CIBIL formula:

  1. Chi-Square Test     — Categorical feature vs Credit Score
  2. Cramér's V          — Categorical-Categorical correlation strength
  3. ANOVA (F-test)      — Numerical feature vs Credit Score groups

Results are saved as CSV reports and printed as formatted tables.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import os
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────
# 1. CHI-SQUARE TEST
# ──────────────────────────────────────────────────────────────

def chi_square_test(df: pd.DataFrame,
                    categorical_cols: list,
                    target_col: str = 'Credit_Score',
                    alpha: float = 0.05) -> pd.DataFrame:
    """
    Test independence between each categorical feature and the target.

    Returns a DataFrame with:
      - Feature name
      - Chi² statistic
      - p-value
      - Degrees of freedom
      - Significant? (Yes/No)
      - Interpretation
    """
    print("\n" + "="*60)
    print("PHASE 2a: CHI-SQUARE TEST (Categorical vs Credit Score)")
    print("="*60)

    results = []

    for col in categorical_cols:
        if col not in df.columns:
            continue
        # Build contingency table
        contingency = pd.crosstab(df[col], df[target_col])
        chi2, p_val, dof, expected = chi2_contingency(contingency)

        # Check minimum expected frequency assumption
        min_exp = expected.min()
        assumption_ok = min_exp >= 5

        significant = p_val < alpha
        interp = (
            f"SIGNIFICANT — {col} is strongly associated with Credit Score"
            if significant else
            f"Not significant — {col} may not predict Credit Score"
        )

        results.append({
            'Feature': col,
            'Chi2_Statistic': round(chi2, 4),
            'p_value': round(p_val, 6),
            'Degrees_of_Freedom': dof,
            'Min_Expected_Freq': round(min_exp, 2),
            'Assumption_OK': assumption_ok,
            'Significant': significant,
            'Interpretation': interp
        })

        sig_str = "✅ SIGNIFICANT" if significant else "❌ Not significant"
        print(f"  {col:<35} χ²={chi2:>10.2f}  p={p_val:.6f}  {sig_str}")

    results_df = pd.DataFrame(results)
    print(f"\n  → {results_df['Significant'].sum()}/{len(results_df)} features are statistically significant (α={alpha})")
    return results_df


# ──────────────────────────────────────────────────────────────
# 2. CRAMÉR'S V  (Categorical-Categorical strength)
# ──────────────────────────────────────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramér's V statistic between two categorical series.
    Returns a value in [0, 1]: 0 = no association, 1 = perfect association.
    """
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2) / (n-1)
    k_corr = k - ((k-1)**2) / (n-1)
    denom = min((k_corr-1), (r_corr-1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2_corr / denom)


def cramers_v_matrix(df: pd.DataFrame,
                     categorical_cols: list,
                     threshold: float = 0.7) -> pd.DataFrame:
    """
    Build a Cramér's V correlation matrix for all categorical features.
    Identifies highly correlated pairs that are redundant.

    Returns the correlation matrix as a DataFrame.
    """
    print("\n" + "="*60)
    print("PHASE 2b: CRAMÉR'S V (Categorical-Categorical Correlation)")
    print("="*60)

    valid_cols = [c for c in categorical_cols if c in df.columns]
    n = len(valid_cols)
    matrix = np.zeros((n, n))

    for i, col1 in enumerate(valid_cols):
        for j, col2 in enumerate(valid_cols):
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                v = cramers_v(df[col1].astype(str), df[col2].astype(str))
                matrix[i][j] = v
                matrix[j][i] = v   # symmetric

    corr_df = pd.DataFrame(matrix, index=valid_cols, columns=valid_cols)

    # Report highly correlated pairs
    print("\n  Cramér's V Correlation Matrix:")
    print(corr_df.round(3).to_string())

    print(f"\n  Pairs with V > {threshold} (potentially redundant):")
    found = False
    for i, col1 in enumerate(valid_cols):
        for j, col2 in enumerate(valid_cols):
            if i < j and matrix[i][j] > threshold:
                print(f"    ⚠  {col1} ↔ {col2}  V={matrix[i][j]:.3f}")
                found = True
    if not found:
        print("    ✅ No highly correlated categorical pairs found.")

    return corr_df


# ──────────────────────────────────────────────────────────────
# 3. ANOVA F-TEST
# ──────────────────────────────────────────────────────────────

def anova_test(df: pd.DataFrame,
               numerical_cols: list,
               target_col: str = 'Credit_Score',
               alpha: float = 0.05) -> pd.DataFrame:
    """
    One-Way ANOVA: test if numerical feature means differ significantly
    across Credit Score groups (Poor=0, Standard=1, Good=2).

    Returns a DataFrame with F-statistic, p-value, and effect size (η²).
    """
    print("\n" + "="*60)
    print("PHASE 2c: ANOVA F-TEST (Numerical Features vs Credit Score Groups)")
    print("="*60)

    target_values = sorted(df[target_col].dropna().unique())
    results = []

    for col in numerical_cols:
        if col not in df.columns:
            continue

        # Build groups: one list of values per credit score class
        groups = []
        group_means = {}
        for val in target_values:
            group_data = df.loc[df[target_col] == val, col].dropna().values
            if len(group_data) > 1:
                groups.append(group_data)
                group_means[int(val)] = round(np.mean(group_data), 2)

        if len(groups) < 2:
            continue

        f_stat, p_val = f_oneway(*groups)

        # Effect size: η² (eta-squared)
        grand_mean = df[col].dropna().mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total   = sum((x - grand_mean)**2 for g in groups for x in g)
        eta_sq     = ss_between / ss_total if ss_total > 0 else 0

        significant = p_val < alpha
        effect_label = (
            "Large"  if eta_sq > 0.14 else
            "Medium" if eta_sq > 0.06 else
            "Small"  if eta_sq > 0.01 else "Negligible"
        )

        results.append({
            'Feature': col,
            'F_Statistic': round(f_stat, 4),
            'p_value': round(p_val, 6),
            'Effect_Size_eta2': round(eta_sq, 4),
            'Effect_Label': effect_label,
            'Mean_Poor': group_means.get(0, np.nan),
            'Mean_Standard': group_means.get(1, np.nan),
            'Mean_Good': group_means.get(2, np.nan),
            'Significant': significant
        })

        sig_str = "✅" if significant else "❌"
        print(f"  {sig_str} {col:<35}  F={f_stat:>10.2f}  p={p_val:.6f}  η²={eta_sq:.4f} ({effect_label})")

    results_df = pd.DataFrame(results).sort_values('F_Statistic', ascending=False)
    print(f"\n  → {results_df['Significant'].sum()}/{len(results_df)} numerical features are significant (α={alpha})")
    print(f"  → Top predictor: {results_df.iloc[0]['Feature']}  (F={results_df.iloc[0]['F_Statistic']:.2f})")
    return results_df


# ──────────────────────────────────────────────────────────────
# MAIN: Run All Statistical Tests
# ──────────────────────────────────────────────────────────────

def run_statistical_tests(X_raw: pd.DataFrame,
                          y: pd.Series,
                          feature_names: list,
                          output_dir: str = 'results/reports') -> dict:
    """
    Run all three statistical tests and save results.

    Parameters
    ----------
    X_raw : pd.DataFrame  — unscaled feature matrix (original values)
    y     : pd.Series     — encoded target (0=Poor, 1=Standard, 2=Good)
    feature_names : list  — all feature column names
    output_dir    : str   — where to save CSV reports

    Returns
    -------
    dict with keys: 'chi2', 'cramers_v', 'anova'
    """
    os.makedirs(output_dir, exist_ok=True)

    # Combine for easy joint analysis
    df = X_raw.copy()
    df['Credit_Score'] = y.values

    # Define which features are categorical vs numerical for testing
    # (Note: after encoding, categorical cols are still ordinal integers
    #  but we treat them as categorical for Chi-Square purposes)
    categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    numerical_cols = [c for c in feature_names if c not in categorical_cols]

    # 1. Chi-Square
    chi2_results = chi_square_test(df, categorical_cols, target_col='Credit_Score')
    chi2_results.to_csv(os.path.join(output_dir, 'chi_square_results.csv'), index=False)
    print(f"\n  💾 Saved: {output_dir}/chi_square_results.csv")

    # 2. Cramér's V
    cramers_results = cramers_v_matrix(df, categorical_cols)
    cramers_results.to_csv(os.path.join(output_dir, 'cramers_v_matrix.csv'))
    print(f"  💾 Saved: {output_dir}/cramers_v_matrix.csv")

    # 3. ANOVA
    anova_results = anova_test(df, numerical_cols, target_col='Credit_Score')
    anova_results.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)
    print(f"  💾 Saved: {output_dir}/anova_results.csv")

    # Print the CIBIL comparison insight
    print("\n" + "="*60)
    print("📊 KEY INSIGHT: Why ML > Traditional CIBIL")
    print("="*60)
    top_5 = anova_results.head(5)['Feature'].tolist()
    cibil_features = {'Num_of_Loan', 'Delay_from_due_date', 'Outstanding_Debt', 'Num_of_Delayed_Payment'}
    new_features = [f for f in top_5 if f not in cibil_features]
    print(f"  Top 5 ANOVA predictors: {top_5}")
    print(f"  Features CIBIL ignores but are significant: {new_features}")

    return {
        'chi2': chi2_results,
        'cramers_v': cramers_results,
        'anova': anova_results
    }


if __name__ == "__main__":
    print("Statistical Tests Module — import and call run_statistical_tests()")
