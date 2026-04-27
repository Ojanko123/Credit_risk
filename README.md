# Credit Risk Model - PD Modelling & Scorecard (Python)

## Overview

An end-to-end credit risk modelling pipeline built on the LendingClub loan dataset. The project implements industry-standard techniques used by banks and financial institutions to assess borrower creditworthiness - including WoE/IV feature selection, logistic regression PD modelling, XGBoost, scorecard scaling, and model stability monitoring via PSI.

This project directly addresses real-world credit risk questions:
- What is the probability that a borrower will default on their loan?
- Which borrower characteristics are most predictive of default?
- How stable is the model across different populations?
- What credit score should be assigned to a given borrower?

## Dataset

- **Source:** [LendingClub Issued Loans — Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
- **Size:** 759,338 loans × 72 columns
- **Period:** 2016–2017
- **Target variable:** `loan_status` → converted to binary (1 = Default, 0 = Fully Paid)

## Tools & Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Mathematical operations, array handling |
|Statsmodels | Fitting a logistic regression with a more statistical output | 
| Scikit-learn | Logistic regression, train/test split, evaluation metrics |
| XGBoost | Gradient boosting model for improved default prediction |
| Matplotlib | All visualizations |
| Seaborn | Confusion matrix heatmaps |
| SciPy | Trimmed mean imputation, Kolmogorov-Smirnov normality test |

## Methodology

### Phase 1 — Data Loading & Exploration
Loaded 759,338 loan records and examined the distribution of loan statuses to understand the composition of the dataset before any modelling decisions.

### Phase 2 — Data Cleaning & Preprocessing

**Column selection:** Reduced 72 columns to 19 features based on two criteria:
- Available at loan application time (no data leakage from post-approval columns)
- Logical business relevance to default prediction

**Target variable definition:**
- 0 = Fully Paid (good borrower)
- 1 = Charged Off / Default / Late 16+ days (bad borrower)
- Removed "Current" loans — outcome unknown, cannot be labelled

**Missing value treatment:**
- Numeric columns → filled with **5% trimmed mean** (removes top and bottom 2.5% of values before averaging — more robust than mean against outliers, more informative than median)
- Categorical columns → filled with mode (most frequent value)

**Text cleaning:** `emp_length` converted from strings ("10+ years") to integers. `int_rate` and `revol_util` stripped of `%` and converted to float.

### Phase 3 — Feature Engineering

Seven new features were created from existing columns to capture relationships the raw variables couldn't express alone:

| Feature | Formula | Business Meaning |
|---|---|---|
| `loan_to_income` | loan_amnt / annual_inc | How large is the loan relative to earnings? |
| `payment_to_income` | installment / (annual_inc/12) | Monthly payment burden relative to income |
| `revol_to_income` | revol_bal / annual_inc | Credit card debt relative to income |
| `has_pub_rec` | pub_rec > 0 → 1 | Any derogatory public record? |
| `has_delinq` | delinq_2yrs > 0 → 1 | Any recent delinquency? |
| `high_inq` | inq_last_6mths > 3 → 1 | Desperately seeking credit? |
| `high_revol_util` | revol_util > 80% → 1 | Maxed out on credit cards? |
| `issue_month` | from issue_d | Seasonality — month of loan issue |
| `issue_quarter` | from issue_d | Seasonality — quarter of loan issue |

### Phase 4 — WoE & Information Value

Weight of Evidence (WoE) transforms each feature to measure how strongly each group of borrowers predicts default vs non-default. Information Value (IV) summarizes total predictive power.

Features with IV < 0.02 were excluded. This step applies only to the Logistic Regression pipeline.

### Phase 5 — Logistic Regression (WoE Features)

WoE encoding applied to all selected features before modelling, enforcing monotonic relationships with default probability — a regulatory requirement in real bank models.

- 80/20 train/test split with stratification
- Evaluated with AUC, ROC curve, and confusion matrix
- Fitted a Logistic regression with a more statistical output.
- **Why this matters:** Most machine learning pipelines only report AUC and accuracy. 
Reporting odds ratios and p-values connects the model to the statistical theory 
underlying logistic regression and mirrors the output expected in regulated 
environments.
- **Example interpretation:** An odds ratio > 1 for a WoE-encoded feature means 
higher values of that feature increase the odds of default. An odds ratio < 1 
means higher values reduce default probability. Features with p-value > 0.05 
are not statistically significant at the 95% confidence level.
 - **AUC: 0.6933**

### Phase 6 — XGBoost (Raw Features)

XGBoost was trained on raw features with Label Encoding instead of WoE. This allows the model to find its own optimal splits rather than being constrained by WoE bins, which is why it outperforms logistic regression on AUC.

Key parameters:
- `n_estimators=300` — 300 sequential decision trees
- `max_depth=5` — depth of each tree
- `scale_pos_weight` — handles class imbalance natively
- `subsample=0.8` — 80% of rows per tree (prevents overfitting)
- `colsample_bytree=0.8` — 80% of features per tree (prevents overfitting)
- **AUC: 0.7100**

### Phase 7 — Model Comparison

Both models evaluated side by side on ROC curve and confusion matrix. XGBoost catches significantly more actual defaulters (higher recall on class 1) at the cost of more false positives — a trade-off that is acceptable in credit risk since missing a defaulter is far more costly than rejecting a good customer.

### Phase 8 — Credit Scorecard Scaling

Logistic regression probabilities converted to a credit score on a 300–850 scale using the industry-standard PDO (Points to Double the Odds) formula:
Score = Offset + Factor × log-odds
Factor = PDO / ln(2)      [PDO = 20]
Offset = Base Score − Factor × ln(Base Odds)   [Base Score = 600]

A Kolmogorov-Smirnov test was applied to verify the distributional properties of the assigned scores.

### Phase 9 — PSI (Model Monitoring)

Population Stability Index compares predicted probability distributions between training and test populations to detect model drift.

## What Changed from Version 1 — and Why

### 1. Missing value imputation: median → 5% trimmed mean
**Why:** Columns like `annual_inc` and `revol_bal` contain extreme outliers (some borrowers report very high incomes). The regular mean is pulled upward by these extremes, producing unrealistic fill values. The trimmed mean removes the top and bottom 2.5% of values before averaging, giving a more representative central estimate without discarding as much information as the median.

### 2. Feature engineering added
**Why:** Raw variables like `loan_amnt` and `annual_inc` carry less signal individually than their ratio (`loan_to_income`). A loan of €50,000 means something very different for someone earning €30,000 vs someone earning €200,000. Creating ratio and flag features explicitly captures these relationships, giving both models richer input to work with.

### 3. XGBoost added alongside Logistic Regression
**Why:** Logistic regression assumes linear relationships between WoE-encoded features and default probability. XGBoost makes no such assumption — it builds decision trees that can capture complex non-linear patterns. Running XGBoost on raw features (without WoE encoding) allows it to find its own optimal splits, which is why it achieves a higher AUC (0.71 vs 0.69).

### 4. Two separate encoding strategies
**Why:** WoE encoding is kept for Logistic Regression because it enforces the monotonic relationships that logistic regression requires and is an industry-standard approach used by real banks and regulators. XGBoost uses Label Encoding on raw features because it is a tree-based model that does not require monotonicity — constraining it with WoE bins actually limits its performance.

## Key Results

| Metric | Value |
|---|---|
| Dataset size | 759,338 loans |
| Default rate | ~30% |
| Logistic Regression AUC (WoE) | 0.6933 |
| XGBoost AUC (Raw features) | 0.7100 |
| Mean credit score | ~540 |
| PSI | < 0.10 (stable) |

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
2. Place the CSV in the same directory as `credit_risk.py`
3. Update the file path in the script
4. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy imbalanced-learn statsmodels
```
5. Run:
```bash
python credit_risk.py
```

## Concepts Demonstrated

- Data leakage prevention in feature selection
- Robust missing value imputation (trimmed mean)
- Feature engineering — ratio features, binary flags, seasonality
- Weight of Evidence (WoE) and Information Value (IV)
- Logistic regression with WoE encoding (industry standard)
- XGBoost with raw features and class imbalance handling
- Model evaluation: AUC, ROC curve, confusion matrix
- Credit scorecard scaling (PDO method)
- Normality testing (KS test) on score distribution
- Population Stability Index (PSI) for model monitoring
- Class imbalance awareness in credit risk context

**Oresti Janko**
BSc Statistics and Insurance Science — University of Piraeus
Focus: Credit risk modelling, statistical analysis, Python
