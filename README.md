# Credit_risk



## Overview

An end-to-end credit risk modelling pipeline built on the LendingClub loan dataset. The project implements industry-standard techniques used by banks and financial institutions to assess borrower creditworthiness — including WoE/IV feature selection, logistic regression PD modelling, scorecard scaling, and model stability monitoring via PSI.

This project directly addresses real-world credit risk questions:
- What is the probability that a borrower will default on their loan?
- Which borrower characteristics are most predictive of default?
- How stable is the model across different populations?
- What credit score should be assigned to a given borrower?

---

## Dataset

- **Source:** [LendingClub Issued Loans — Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
- **Size:** 759,338 loans × 72 columns
- **Period:** 2016–2017
- **Target variable:** `loan_status` → converted to binary (1 = Default, 0 = Fully Paid)

---

## Tools & Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Mathematical operations, array handling |
|Statsmodels| Logistic regression with a more statistical output |
| Scikit-learn | Logistic regression, train/test split, evaluation metrics |
| Matplotlib | All visualizations |
| Seaborn | Confusion matrix heatmap |
| SciPy | Kolmogorov-Smirnov normality test on score distribution |

---

---

## Methodology

### Phase 1 — Data Loading & Exploration
Loaded 759,338 loan records and examined the distribution of loan statuses to understand the composition of the dataset before any modelling decisions.

### Phase 2 — Data Cleaning & Preprocessing

**Column selection:** Reduced 72 columns to 17 features based on two criteria:
- Available at loan application time (no data leakage from post-approval columns)
- Logical business relevance to default prediction

Key features retained: `loan_amnt`, `int_rate`, `grade`, `dti`, `annual_inc`, `emp_length`, `home_ownership`, `revol_util`, `delinq_2yrs`, `inq_last_6mths` and others.

**Target variable definition:**
- 0 = Fully Paid (good borrower)
- 1 = Charged Off / Default / Late 16+ days (bad borrower)
- Removed "Current" loans — outcome unknown, cannot be labelled

**Missing value treatment:**
- Numeric columns → filled with median (robust to outliers)
- Categorical columns → filled with mode (most frequent value)

**Text cleaning:** `emp_length` converted from strings ("10+ years") to integers (10). `int_rate` and `revol_util` stripped of `%` and converted to float.

### Phase 3 — WoE & Information Value

**Weight of Evidence (WoE)** transforms each feature to measure how strongly each group of borrowers predicts default vs non-default:

```
WoE = ln(Distribution of Events / Distribution of Non-Events)
```

**Information Value (IV)** summarizes the total predictive power of each feature:

| IV Range | Interpretation |
|---|---|
| < 0.02 | Useless — dropped |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| 0.3 – 0.5 | Strong |
| > 0.5 | Suspicious |

Features with IV < 0.02 were excluded. WoE encoding was applied to all selected features before modelling, replacing raw values with their WoE scores to enforce monotonic relationships with default probability — a regulatory requirement in real bank models.

### Phase 4 — Logistic Regression (PD Model)

- 80/20 train/test split with stratification to preserve default rate balance
- Logistic regression fitted on WoE-encoded features
- Model evaluated with AUC/ROC curve, confusion matrix, and classification report

**Key metrics:**
- AUC score measures the model's ability to rank defaulters above non-defaulters
- Confusion matrix breaks down True Positives, False Positives, True Negatives, False Negatives
- In credit risk, False Negatives (missed defaults) are the most costly outcome

### Phase 5 — Scorecard Scaling

Predicted probabilities converted to a credit score on a 300–850 scale using the industry-standard PDO (Points to Double the Odds) formula:

```
Score = Offset + Factor × log-odds
Factor = PDO / ln(2)
Offset = Base Score − Factor × ln(Base Odds)
```

Parameters used: PDO = 20, Base Score = 600, Base Odds = 1/19

A Kolmogorov-Smirnov test was applied to the score distribution to formally test whether scores follow a normal distribution.

### Phase 6 — PSI (Model Monitoring)

Population Stability Index compares the distribution of predicted default probabilities between training and test populations:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

| PSI | Interpretation |
|---|---|
| < 0.10 | Stable — no action needed |
| 0.10 – 0.20 | Moderate drift — monitor |
| > 0.20 | Significant shift — retrain model |

PSI is a critical component of model risk management in real bank deployments — it detects when changing economic conditions have made the model unreliable.

---

## Key Findings

- **Default rate:** ~30% of completed loans resulted in default or serious delinquency
- **Most predictive features:** `grade`, `sub_grade`, `int_rate` had the highest IV scores — consistent with their role as LendingClub's own risk assessment
- **Model AUC:** Evaluated on held-out test set
- **Score distribution:** KS test applied to verify distributional properties of assigned credit scores
- **PSI:** Compared train vs test population stability to validate model robustness

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans)
2. Place the CSV file in the same directory as `credit_risk.py`
3. Update the file path in the script
4. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```
5. Run:
```bash
python credit_risk.py
```

---

## Concepts Demonstrated

- Data leakage prevention in feature selection
- Weight of Evidence (WoE) and Information Value (IV) for feature engineering
- Binary classification with logistic regression
- Model evaluation: AUC, ROC curve, confusion matrix
- Credit scorecard scaling (PDO method) — industry standard
- Population Stability Index (PSI) for model monitoring
- Normality testing (KS test) on score distribution
- Class imbalance awareness in credit risk context

---

## Author

**Oresti Janko**
BSc Statistics and Insurance Science — University of Piraeus
Focus: Credit risk modelling, statistical analysis, Python, SQL
