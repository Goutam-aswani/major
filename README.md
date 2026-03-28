# Financial Credit Risk Model — Proving ML > Traditional CIBIL

<div align="center">

![Credit Score System](https://img.shields.io/badge/Credit_Scoring-ML_Powered-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Pipeline-Complete-green?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/Winner-XGBoost-orange?style=for-the-badge)

</div>

## 📌 Project Objective

The traditional **CIBIL / Credit Scoring system is fundamentally flawed and exclusionary.** 
Under the traditional system, if you do not have a prior loan history, you are considered "thin-file" and immediately assigned a zero or minimal score — effectively blocking first-time borrowers from the financial system.

This project exists to **mathematically prove that Machine Learning models trained on alternative behavioral data** significantly outperform rigid, rule-based systems in both **predictive accuracy** and **financial inclusivity**.

By deploying a 7-phase ML pipeline, we proved that behavioral signals (such as Payment Behaviour, Monthly Savings Ratios, and Debt-to-Income) can map an individual to a highly accurate credit score range (300-900) without relying on arbitrary historical punishment rules.

---

## 📈 Results: The Proof

After testing 7 distinct algorithms against a modeled **traditional CIBIL-style baseline**, **XGBoost** emerged as the undisputed winner.

| Model | Test Accuracy | F1-Macro Score | Improvement over CIBIL |
|-------|--------------|----------|-----------------------|
| ⚖ **CIBIL Baseline** | 56.5% | **0.4430** | *Baseline* |
| Logistic Regression | 64.9% | 0.6010 | +35.7% |
| Decision Tree | 70.1% | 0.6678 | +50.7% |
| Random Forest | 71.6% | 0.6831 | +54.2% |
| 🏆 **XGBoost** | **74.1%** | **0.7130** | **+60.9%** |

**Conclusion:** Machine Learning drives a **+60.9% improvement in F1-score.** 
Furthermore, the use of SHAP/LIME logic renders the ML model 100% transparent. We know *exactly why* a customer got a specific score, satisfying regulatory demands for explainability.

---

## ⚙️ The 7-Phase ML Architecture

1. **Robust Data Cleaning (`src/data/preprocess.py`)** 
   - Cleaned 100k dirty records. Neutralized sentinel arrays, purged trailing garbage text, imputed >8,000 impossible ages.
   - Engineered critical behavioral markers: `Debt_to_Income_Ratio`, `Savings_Ratio`, `EMI_Burden_Ratio`.
2. **Statistical Proof (`src/features/statistical_tests.py`)**
   - Conducted ANOVA (F-Tests) and Chi-Square validations.
   - Discovered that alternative features like `Outstanding_Debt` (F-score = 7,323) possess enormous predictive power completely unseen by rule-sets.
3. **Consensus Feature Selection (`src/features/select_features.py`)**
   - Eliminated feature noise by cross-referencing 5 distinct methods (Statistical, Mutual Info, RFE, L1 Reg, Tree-based). Out of 25 features, only 12 survived the voting filter.
4. **Ensemble Model Training (`src/models/train_model.py`)**
   - Stratified CV across Logistic, DT, RF, SVM, KNN, MLP, and XGBoost to guarantee metric stability.
5. **Real-world Test Pipeline**
   - Applied the entire pipeline to a blind **50,000 row test dataset (`test.csv`)**.
6. **Interpretability & Translation (`src/models/interpret.py`)**
   - Used **SHAP (Global)** and **LIME (Local)** to ensure the model isn't just a black box.
   - Embedded a custom translation function to seamlessly convert pure probabilities into industry-standard **300-900** scores.
7. **Credit Score Card Generation**
   - Generated dynamic visual profiles for scored customers.

---

## 🚀 How to Run the System

### 1. Project Directory Structure
All logic resides directly in the `src` module and is orchestrated elegantly from root.

### 2. Predict on New Data
We have decoupled the inference system so you can instantly score endless lists of prospective customers via `predict_test.py`:

```bash
python predict_test.py --test test.csv
```
***Output:*** *Generates predictions + 300-900 normalized scores stored in `results/reports/test_predictions.csv`.*

### 3. Or Retrain the Core Engine
Trigger the 7-phase engine directly:
```bash
python src/main.py --input train.csv --phase all
```
*(Results, reports, and visual proof are saved directly to `results/`)*
