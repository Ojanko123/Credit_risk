# CREDIT RISK MODEL - LendingClub Dataset (Final Version)
# Methodology:
# Logistic Regression: WoE encoded features (industry standard)
# XGBoost: Raw features with Label Encoding (better performance)
#
# Phases:
# 1. Data Loading & Exploration
# 2. Data Cleaning & Preprocessing
# 3. Feature Engineering
# 4. WoE/IV Analysis & Feature Selection
# 5. Logistic Regression PD Model (WoE features)
# 6. XGBoost PD Model (Raw features)
# 7. Model Comparison
# 8. Credit Scorecard Scaling (Logistic Regression)
# 9. PSI - Model Stability Monitoring
#####################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy.stats import trim_mean
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#########################################
# PHASE 1 - DATA LOADING & EXPLORATION
#########################################
print("PHASE 1 - DATA LOADING & EXPLORATION")

loans = pd.read_csv("C:\\Users\\ojank\\Desktop\\SQL\\lc_2016_2017.csv",
                    low_memory=False)

print(f"Raw data shape: {loans.shape}")
print("\nLoan Status value counts:")
print(loans['loan_status'].value_counts())

############################################
# PHASE 2 - DATA CLEANING & PREPROCESSING
############################################
print("PHASE 2 - DATA CLEANING & PREPROCESSING")

# Step 1: Keep only relevant columns
#Keeping all the 72 columns at this point is not helpful so I will try to keep those with predictive value to my model  
# Each kept column has a clear reason to influence default probability. 
# No post-approval columns (no data leakage)
cols_to_keep = [
    'loan_amnt',             # Loan amount requested
    'int_rate',              # Interest rate
    'grade',                 # LendingClub risk grade (A-G)
    'sub_grade',             # LendingClub sub grade (A1-G5)
    'emp_length',            # Employment length
    'home_ownership',        # Housing status
    'annual_inc',            # Annual income
    'verification_status',   # Income verification status
    'purpose',               # Loan purpose
    'dti',                   # Debt-to-income ratio
    'delinq_2yrs',           # Delinquencies in last 2 years
    'inq_last_6mths',        # Credit inquiries last 6 months
    'open_acc',              # Number of open credit lines
    'pub_rec',               # Public derogatory records
    'revol_bal',             # Revolving balance
    'revol_util',            # Revolving utilization rate
    'total_acc',             # Total credit lines ever
    'installment',           # Monthly installment (for ratio feature)
    'issue_d',               # Issue date (for seasonality features)
    'loan_status'            # Target variable
]

cols_to_keep = [c for c in cols_to_keep if c in loans.columns]
loans = loans[cols_to_keep].copy()
print(f"Shape after column selection: {loans.shape}")

# Step 2: Define target variable
# Keep only loans with a known final outcome
# Remove 'Current' loans — outcome unknown
loans = loans[loans['loan_status'].isin([
    'Fully Paid', 'Charged Off', 'Default',
    'Late (31-120 days)', 'Late (16-30 days)',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid'
])]

# 0 = No Default (Fully Paid)
# 1 = Default (Charged Off, Late, etc.)
loans['target'] = np.where(
    loans['loan_status'].isin(['Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid']), 0, 1)

print(f"\nTarget distribution:")
print(loans['target'].value_counts())
print(f"Default rate: {loans['target'].mean():.2%}")

loans.drop('loan_status', axis=1, inplace=True)

# Step 3: Handle missing values
print("\nMissing values before cleaning:")
print(loans.isnull().sum())

# Numeric: 5% trimmed mean
# Removes top and bottom 2.5% of values before averaging
# More robust than mean against outliers, more informative than median
numeric_cols = loans.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    trimmed = trim_mean(loans[col].dropna(), proportiontocut=0.025)
    loans[col] = loans[col].fillna(trimmed)

# Categorical: mode (most frequent value)
categorical_cols = loans.select_dtypes(include=['object']).columns
for col in categorical_cols:
    loans[col] = loans[col].fillna(loans[col].mode()[0])

print("\nMissing values after cleaning:")
print(loans.isnull().sum())

# Step 4: Clean emp_length 
if loans['emp_length'].dtype == object:
    loans['emp_length'] = loans['emp_length'].str.replace(' years', '')
    loans['emp_length'] = loans['emp_length'].str.replace(' year', '')
    loans['emp_length'] = loans['emp_length'].str.replace('< 1', '0')
    loans['emp_length'] = loans['emp_length'].str.replace('10+', '10')
    loans['emp_length'] = pd.to_numeric(loans['emp_length'], errors='coerce')
    loans['emp_length'] = loans['emp_length'].fillna(loans['emp_length'].median())
else:
    print("emp_length is already numeric — skipping string cleaning")

# Step 5: Clean int_rate 
if loans['int_rate'].dtype == object:
    loans['int_rate'] = loans['int_rate'].str.replace('%', '').astype(float)

# Step 6: Clean revol_util 
if loans['revol_util'].dtype == object:
    loans['revol_util'] = loans['revol_util'].str.replace('%', '').astype(float)
    loans['revol_util'] = loans['revol_util'].fillna(loans['revol_util'].median())

print("\nData types after cleaning:")
print(loans.dtypes)

################################
# PHASE 3 - FEATURE ENGINEERING
################################
print("PHASE 3 — FEATURE ENGINEERING")

# Ratio features
# Loan to income: how large is the loan relative to annual earnings?
# High ratio = borrower is overextended
loans['loan_to_income'] = loans['loan_amnt'] / (loans['annual_inc'] + 1)

# Payment to income: monthly installment as % of monthly income
# High ratio = borrower is financially stretched month to month
if 'installment' in loans.columns:
    loans['payment_to_income'] = loans['installment'] / (loans['annual_inc'] / 12 + 1)
    loans.drop('installment', axis=1, inplace=True)

# Revolving balance to income: credit card debt relative to income
loans['revol_to_income'] = loans['revol_bal'] / (loans['annual_inc'] + 1)

# Binary flag features 
# Has any public derogatory record (bankruptcy, judgement etc.)?
loans['has_pub_rec'] = (loans['pub_rec'] > 0).astype(int)

# Had any delinquency in the last 2 years?
loans['has_delinq'] = (loans['delinq_2yrs'] > 0).astype(int)

# High credit inquiry pressure (>3 = desperately seeking credit)
loans['high_inq'] = (loans['inq_last_6mths'] > 3).astype(int)

# High revolving utilization (>80% = maxed out on credit cards)
loans['high_revol_util'] = (loans['revol_util'] > 80).astype(int)

# Seasonality features from issue date 
if 'issue_d' in loans.columns:
    try:
        loans['issue_d']       = pd.to_datetime(loans['issue_d'])
        loans['issue_month']   = loans['issue_d'].dt.month
        loans['issue_quarter'] = loans['issue_d'].dt.quarter
        loans.drop('issue_d', axis=1, inplace=True)
        print("Seasonality features created: issue_month, issue_quarter")
    except:
        loans.drop('issue_d', axis=1, inplace=True)
        print("Could not parse issue_d — skipping seasonality features")

print(f"\nNew features: loan_to_income, payment_to_income, revol_to_income,")
print(f"              has_pub_rec, has_delinq, high_inq, high_revol_util,")
print(f"              issue_month, issue_quarter")
print(f"\nShape after feature engineering: {loans.shape}")

# =============================================================
# PHASE 4 - WoE & INFORMATION VALUE
# =============================================================
print("\n" + "=" * 60)
print("PHASE 4 — WoE & INFORMATION VALUE")
print("=" * 60)

def calculate_woe_iv(df, feature, target, bins=10):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV).

    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = Sum((Dist Events - Dist Non-Events) * WoE)

    IV Interpretation:
    < 0.02  : Useless — dropped
    0.02-0.1: Weak
    0.1-0.3 : Medium
    0.3-0.5 : Strong
    > 0.5   : Suspicious (possible data leakage)
    """
    df = df[[feature, target]].copy()

    if df[feature].dtype in [np.float64, np.int64]:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        df['bin'] = df[feature]

    grouped = df.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    total_events     = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    grouped['dist_events']     = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events

    grouped['dist_events']     = grouped['dist_events'].replace(0, 0.0001)
    grouped['dist_non_events'] = grouped['dist_non_events'].replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['iv']  = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['woe']

    iv = grouped['iv'].sum()
    return grouped['woe'], iv

# Calculate IV for all features
print("\nCalculating Information Values...")
iv_results = {}
features = [col for col in loans.columns if col != 'target']

for feature in features:
    try:
        _, iv = calculate_woe_iv(loans, feature, 'target')
        iv_results[feature] = iv
    except Exception as e:
        print(f"Could not compute IV for {feature}: {e}")

iv_df = pd.DataFrame.from_dict(iv_results, orient='index', columns=['IV'])
iv_df = iv_df.sort_values('IV', ascending=False)
print("\nInformation Values:")
print(iv_df.to_string())

# Keep features with IV > 0.02
selected_features = iv_df[iv_df['IV'] > 0.02].index.tolist()
print(f"\nSelected {len(selected_features)} features with IV > 0.02")

# IV Chart
plt.figure(figsize=(10, 8))
iv_df[iv_df['IV'] > 0.02]['IV'].sort_values().plot(
    kind='barh', color='steelblue', edgecolor='black', alpha=0.8)
plt.title('Information Value by Feature\n(includes engineered features)',
          fontsize=14)
plt.xlabel('Information Value')
plt.tight_layout()
plt.savefig('iv_chart.png', dpi=150)
plt.show()


#######################################################
# PHASE 5 - LOGISTIC REGRESSION (WoE Encoded Features)
#######################################################
print("PHASE 5 — LOGISTIC REGRESSION (WoE Features)")

# WoE encode selected features
loans_woe = pd.DataFrame()

for feature in selected_features:
    try:
        woe_map, _ = calculate_woe_iv(loans, feature, 'target')
        if loans[feature].dtype in [np.float64, np.int64]:
            bins = pd.qcut(loans[feature], q=10, duplicates='drop', retbins=False)
            loans_woe[feature + '_woe'] = bins.map(woe_map)
        else:
            loans_woe[feature + '_woe'] = loans[feature].map(
                loans.groupby(feature)['target'].apply(
                    lambda x: np.log((x.mean() + 0.0001) / (1 - x.mean() + 0.0001))
                )
            )
    except Exception as e:
        print(f"Skipping {feature}: {e}")

loans_woe['target'] = loans['target'].values

# Convert all to numeric and fill NaNs
for col in loans_woe.columns:
    loans_woe[col] = pd.to_numeric(loans_woe[col], errors='coerce')
loans_woe = loans_woe.fillna(0)
loans_woe = loans_woe.dropna()

print(f"\nWoE encoded dataset shape: {loans_woe.shape}")

X_woe = loans_woe.drop('target', axis=1)
y_woe = loans_woe['target']

X_train_woe, X_test_woe, y_train_woe, y_test_woe = train_test_split(
    X_woe, y_woe, test_size=0.2, random_state=42, stratify=y_woe)

print(f"Training set: {X_train_woe.shape}, Test set: {X_test_woe.shape}")

# Fit Logistic Regression 
lr = LogisticRegression(max_iter=1000, random_state=42, solver='saga')
lr.fit(X_train_woe, y_train_woe)

lr_probs = lr.predict_proba(X_test_woe)[:, 1]
lr_preds = lr.predict(X_test_woe)
lr_auc   = roc_auc_score(y_test_woe, lr_probs)

print(f"\nLogistic Regression AUC: {lr_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_woe, lr_preds))
# Statsmodels summary (statistical output)
# This gives us the R-style output: coefficients, std errors,
# z-scores, p-values and Pseudo R² — same as glm() in R


X_train_sm = sm.add_constant(X_train_woe)  # adds intercept term
logit_model = sm.Logit(y_train_woe, X_train_sm)
result = logit_model.fit(method='lbfgs', maxiter=1000)
print(result.summary())

# Odds Ratios
# Exponentiate the coefficients to get odds ratios
# An odds ratio > 1 means the feature increases default probability
# An odds ratio < 1 means the feature decreases default probability
odds_ratios = pd.DataFrame({
    'Coefficient': result.params,
    'Odds Ratio':  np.exp(result.params),
    'P-value':     result.pvalues
}).drop('const')

print("\nOdds Ratios:")
print(odds_ratios.sort_values('Odds Ratio', ascending=False).to_string())

#  Pseudo R^2
print(f"\nMcFadden Pseudo R^2:  {result.prsquared:.4f}")
print(f"Log-Likelihood:      {result.llf:.2f}")
print(f"AIC:                 {result.aic:.2f}")
print(f"BIC:                 {result.bic:.2f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test_woe, lr_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title(f'Confusion Matrix — Logistic Regression (AUC={lr_auc:.4f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_lr.png', dpi=150)
plt.show()

#######################################################
# PHASE 6 — XGBoost (Raw Features with Label Encoding)
#######################################################
print("PHASE 6 — XGBoost (Raw Features)")


# Label encode categorical columns for XGBoost 
# Label Encoding converts categories to numbers:
# e.g. grade: A=0, B=1, C=2, D=3, E=4, F=5, G=6
# We let XGBoost find its own splits — more powerful than WoE bins
loans_raw = loans.copy()
le = LabelEncoder()

categorical_cols = loans_raw.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    loans_raw[col] = le.fit_transform(loans_raw[col].astype(str))

X_raw = loans_raw.drop('target', axis=1).fillna(0)
y_raw = loans_raw['target']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

print(f"Training set: {X_train_raw.shape}, Test set: {X_test_raw.shape}")

# Fit XGBoost
neg_count = (y_train_raw == 0).sum()
pos_count = (y_train_raw == 1).sum()
scale     = neg_count / pos_count
print(f"scale_pos_weight = {scale:.2f}")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)
xgb.fit(X_train_raw, y_train_raw)

xgb_probs = xgb.predict_proba(X_test_raw)[:, 1]
xgb_preds = xgb.predict(X_test_raw)
xgb_auc   = roc_auc_score(y_test_raw, xgb_probs)

print(f"\nXGBoost AUC: {xgb_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_raw, xgb_preds))

# Confusion Matrix XGBoost
cm_xgb = confusion_matrix(y_test_raw, xgb_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title(f'Confusion Matrix — XGBoost (AUC={xgb_auc:.4f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_xgb.png', dpi=150)
plt.show()
print("Chart saved: confusion_matrix_xgb.png")

# XGBoost Feature Importance
importances = pd.Series(
    xgb.feature_importances_,
    index=X_raw.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 10))
importances.plot(kind='barh', color='steelblue',
                 edgecolor='black', alpha=0.8)
plt.title('XGBoost Feature Importance', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=150)
plt.show()
print("Chart saved: xgb_feature_importance.png")

#################################
# PHASE 7 - MODEL COMPARISON
#################################
print("\n" + "=" * 60)
print("PHASE 7 — MODEL COMPARISON")
print("=" * 60)

# Combined ROC Curve 
fpr_lr,  tpr_lr,  _ = roc_curve(y_test_woe, lr_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_raw, xgb_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr,  tpr_lr,  color='steelblue', linewidth=2,
         label=f'Logistic Regression — WoE (AUC={lr_auc:.4f})')
plt.plot(fpr_xgb, tpr_xgb, color='red', linewidth=2,
         label=f'XGBoost — Raw Features (AUC={xgb_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — PD Model Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print("Chart saved: roc_curve.png")

print(f"\nLogistic Regression AUC (WoE):   {lr_auc:.4f}")
print(f"XGBoost AUC (Raw features):       {xgb_auc:.4f}")
winner = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
print(f"Best model: {winner}")

############################################################
# PHASE 8 - CREDIT SCORECARD SCALING (Logistic Regression)
############################################################
print("\n" + "=" * 60)
print("PHASE 8 — CREDIT SCORECARD SCALING")
print("=" * 60)

# Convert predicted probabilities to credit scores (300-850 scale)
# Industry-standard PDO (Points to Double the Odds) formula:
# Score = Offset + Factor * log-odds
# Factor = PDO / ln(2)
# Offset = Base Score - Factor * ln(Base Odds)

pdo        = 20          # Points to double the odds
base_score = 600         # Score at base odds
base_odds  = 1/19        # ~5% default rate

factor = pdo / np.log(2)
offset = base_score - factor * np.log(base_odds)

log_odds = np.log(lr_probs / (1 - lr_probs + 1e-10))
scores   = offset + factor * log_odds
scores   = np.clip(scores, 300, 850)

print(f"\nCredit Score Distribution (Logistic Regression):")
print(f"Min score:  {scores.min():.0f}")
print(f"Max score:  {scores.max():.0f}")
print(f"Mean score: {scores.mean():.0f}")


# Score distribution plot
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=50, color='steelblue',
         edgecolor='black', alpha=0.7)
plt.title('Credit Score Distribution (300-850 Scale)', fontsize=14)
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=150)
plt.show()
# Noticing from the plot that score distribution might be the normal one. Running a ks-test to do a hypothesis test


# Kolmogorov-Smirnov test on score distribution
ks_stat, ks_p = stats.kstest(scores, 'norm',
                              args=(scores.mean(), scores.std()))
print(f"\nKolmogorov-Smirnov Test on Score Distribution:")
print(f"KS Statistic: {ks_stat:.4f}")
print(f"P-value:      {ks_p:.4f}")
if ks_p > 0.05:
    print("Scores follow a normal distribution (fail to reject H0)")
else:
    print("Scores do not follow a normal distribution (reject H0)")

# Average score by default status
scores_df = pd.DataFrame({'score': scores, 'default': y_test_woe.values})
print("\nAverage score by default status:")
print(scores_df.groupby('default')['score'].mean())

##############################################
# PHASE 9 — PSI (POPULATION STABILITY INDEX)
##############################################
print("PHASE 9 — PSI (MODEL STABILITY MONITORING)")

def calculate_psi(expected, actual, bins=10):
    """
    PSI measures how much the distribution of predicted probabilities
    has shifted between training and test populations.

    PSI < 0.1  : Stable — no action needed
    PSI 0.1-0.2: Moderate drift — monitor
    PSI > 0.2  : Significant shift — retrain model
    """
    breakpoints = np.linspace(0, 1, bins + 1)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct   = actual_counts   / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 0.0001, actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi        = psi_values.sum()
    return psi, psi_values

# Compare train vs test PD distribution (Logistic Regression)
train_probs_lr = lr.predict_proba(X_train_woe)[:, 1]
test_probs_lr  = lr.predict_proba(X_test_woe)[:, 1]

psi_score, psi_bins = calculate_psi(train_probs_lr, test_probs_lr)

print(f"\nPSI Result (Logistic Regression):")
print(f"PSI = {psi_score:.4f}")
if psi_score < 0.1:
    print("Interpretation: No significant population shift — model is stable")
elif psi_score < 0.2:
    print("Interpretation: Moderate shift — monitor the model closely")
else:
    print("Interpretation: Significant shift — consider retraining the model")

# PSI plot
plt.figure(figsize=(10, 5))
bins_labels = [f"{i*10}-{(i+1)*10}%" for i in range(len(psi_bins))]
plt.bar(bins_labels, psi_bins, color='steelblue',
        edgecolor='black', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.8)
plt.title(f'PSI by PD Bucket (Total PSI = {psi_score:.4f})', fontsize=13)
plt.xlabel('PD Bucket')
plt.ylabel('PSI Contribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('psi_chart.png', dpi=150)
plt.show()
