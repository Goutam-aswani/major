"""
Main Pipeline Orchestrator — ML Credit Scoring System
======================================================
Runs the complete 7-phase pipeline end-to-end:

  Phase 1: Data Cleaning + Feature Engineering
  Phase 2: Statistical Testing (Chi-Square, Cramér's V, ANOVA)
  Phase 3: Feature Selection (5 Methods + Vote Table)
  Phase 4: Model Training (7 Algorithms + CIBIL Baseline)
  Phase 5: Evaluation & Comparison
  Phase 6: Interpretability (SHAP + LIME + Score Cards)
  Phase 7: Final Report

Usage:
  python src/main.py
  python src/main.py --input train.csv
"""

import os
import sys
import argparse
import time
import yaml

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
import pandas as pd

from src.data.preprocess import run_preprocessing
from src.features.statistical_tests import run_statistical_tests
from src.features.select_features import run_feature_selection
from src.models.train_model import run_training
from src.models.interpret import run_interpretability


# ──────────────────────────────────────────────────────────────
# Load config
# ──────────────────────────────────────────────────────────────

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ML Credit Scoring Pipeline')
    parser.add_argument('--input',  default='train.csv',
                        help='Path to raw CSV (default: train.csv)')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--phase',  default='all',
                        choices=['all', 'preprocess', 'stats', 'features',
                                 'train', 'interpret'],
                        help='Which phase to run')
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = cfg.get('random_state', 42)

    # Output directories
    results_dir  = cfg['output']['results_path']
    figures_dir  = cfg['output']['figures_path']
    models_dir   = cfg['output']['models_path']
    reports_dir  = cfg['output']['reports_path']
    data_proc_dir = cfg['data']['processed_data_path']

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(data_proc_dir, exist_ok=True)

    t_start = time.time()

    print("\n" + "█"*60)
    print("  ML-POWERED CREDIT SCORING SYSTEM")
    print("  Proving ML > Traditional CIBIL Score")
    print("█"*60)

    # ─────────────────────────────────────────────────────────
    # PHASE 1: Data Cleaning + Feature Engineering
    # ─────────────────────────────────────────────────────────
    if args.phase in ('all', 'preprocess'):
        prep_results = run_preprocessing(
            raw_csv_path=args.input,
            output_dir=data_proc_dir
        )
    else:
        # Load from saved processed files
        print("\n  Loading previously processed data...")
        X = pd.read_csv(os.path.join(data_proc_dir, 'X_processed.csv'))
        y = pd.read_csv(os.path.join(data_proc_dir, 'y_processed.csv')).squeeze()
        X_raw = pd.read_csv(os.path.join(data_proc_dir, 'X_raw.csv'))
        prep_results = {
            'X': X, 'y': y, 'X_raw': X_raw,
            'feature_names': X.columns.tolist(),
            'target_map_inv': {0: 'Poor', 1: 'Standard', 2: 'Good'}
        }

    X            = prep_results['X']
    y            = prep_results['y']
    X_raw        = prep_results['X_raw']
    feature_names = prep_results['feature_names']
    target_map_inv = prep_results.get('target_map_inv', {0: 'Poor', 1: 'Standard', 2: 'Good'})

    # Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y)

    X_raw_train = X_raw.loc[X_train.index]

    print(f"\n  Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"  Class distribution (train): {y_train.value_counts().to_dict()}")

    if args.phase == 'preprocess':
        print("\n✅ Phase 1 complete. Exiting.")
        return

    # ─────────────────────────────────────────────────────────
    # PHASE 2: Statistical Testing
    # ─────────────────────────────────────────────────────────
    if args.phase in ('all', 'stats'):
        stat_results = run_statistical_tests(
            X_raw=X_raw_train,
            y=y_train,
            feature_names=feature_names,
            output_dir=reports_dir
        )

    if args.phase == 'stats':
        print("\n✅ Phase 2 complete. Exiting.")
        return

    # ─────────────────────────────────────────────────────────
    # PHASE 3: Feature Selection
    # ─────────────────────────────────────────────────────────
    if args.phase in ('all', 'features'):
        top_k = cfg['feature_selection']['top_k_features']
        feat_results = run_feature_selection(
            X=X_train,
            y=y_train,
            k=top_k,
            output_dir=reports_dir
        )
        best_features = feat_results['best_features']
    else:
        # Fall back to all features
        best_features = feature_names

    if args.phase == 'features':
        print("\n✅ Phase 3 complete. Exiting.")
        return

    # ─────────────────────────────────────────────────────────
    # PHASE 4 & 5: Training + Evaluation
    # ─────────────────────────────────────────────────────────
    if args.phase in ('all', 'train'):
        train_results = run_training(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            best_features=best_features,
            output_dir=models_dir,
            reports_dir=reports_dir
        )
        best_model_name = train_results['best_model_name']
        best_model      = train_results['fitted_models'][best_model_name]
    else:
        print("  ⚠ Skipping training phase (load pre-trained model for interpret)")
        return

    if args.phase == 'train':
        print("\n✅ Phases 4-5 complete. Exiting.")
        return

    # ─────────────────────────────────────────────────────────
    # PHASE 6: Interpretability
    # ─────────────────────────────────────────────────────────
    if args.phase in ('all', 'interpret'):
        # Use best_features subset for test data
        X_test_sel  = X_test[best_features]
        X_train_sel = X_train[best_features]

        interp_results = run_interpretability(
            model=best_model,
            X_train=X_train_sel,
            X_test=X_test_sel,
            y_test=y_test,
            feature_names=best_features,
            target_map_inv=target_map_inv,
            figures_dir=figures_dir,
            reports_dir=reports_dir
        )

    # ─────────────────────────────────────────────────────────
    # PHASE 7: Final Summary
    # ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    comparison = train_results['comparison_table']

    print("\n" + "█"*60)
    print("  PIPELINE COMPLETE — FINAL SUMMARY")
    print("█"*60)

    # Print top 3 models
    top3 = comparison.head(4)
    for _, row in top3.iterrows():
        print(f"  {row['Model']:<35}  F1={row['Test_F1_Macro']:.4f}  "
              f"Acc={row['Test_Accuracy']:.4f}")

    print(f"\n  🏆 Best Model  : {best_model_name}")
    print(f"  ⏱  Total Time  : {elapsed/60:.1f} minutes")
    print(f"  📁 Results in  : {results_dir}")
    print("\n  Key Output Files:")
    print(f"    {reports_dir}/chi_square_results.csv")
    print(f"    {reports_dir}/anova_results.csv")
    print(f"    {reports_dir}/feature_selection_votes.csv")
    print(f"    {reports_dir}/model_comparison.csv")
    print(f"    {reports_dir}/credit_score_card.csv")
    print(f"    {figures_dir}/shap_global_importance.png")


if __name__ == "__main__":
    main()
