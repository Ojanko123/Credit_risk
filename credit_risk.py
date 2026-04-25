#Credit risk
####Importing the libraries I am going to use####
import pandas as pd 
import statsmodels.api as sm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')
########################################################################

loans =  pd.read_csv("C:\\Users\\ojank\\Desktop\\SQL\\lc_2016_2017.csv" , low_memory = False)
# After importing the dataset I want to look at the shape of my data.
print("Shape:", loans.shape)        
print("\nLoan Status value counts:")
print(loans['loan_status'].value_counts())
#Keeping all the 72 columns at this point is not helpful so I will try to keep those with predictive value to my model  
#  Each kept column has a clear reason to influence default probability. 
cols_to_keep = [
    'loan_amnt',        # Loan amount requested
    'int_rate',         # Interest rate
    'grade',            # LendingClub risk grade
    'sub_grade',        # LendingClub sub grade
    'emp_length',       # Employment length
    'home_ownership',   # Housing status
    'annual_inc',       # Annual income
    'verification_status', # Income verification
    'purpose',          # Loan purpose
    'dti',              # Debt-to-income ratio
    'delinq_2yrs',      # Delinquencies in last 2 years
    'inq_last_6mths',   # Credit inquiries last 6 months
    'open_acc',         # Number of open credit lines
    'pub_rec',          # Public derogatory records
    'revol_bal',        # Revolving balance
    'revol_util',       # Revolving utilization rate
    'total_acc',        # Total credit lines
    'loan_status'       # Target variable
]
loans = loans[cols_to_keep].copy()
print("\nShape after column selection:", loans.shape)
# --- Step 2: Define target variable ---
# 1 = Default, 0 = No Default
# We keep only loans that have a final status (remove 'Current') ,
# We remove "Current" loans (in order to train our model) because their outcome is unknown — they haven't finished yet, so we don't know if they'll repay or default.
loans = loans[loans['loan_status'].isin([
    'Fully Paid', 'Charged Off', 'Default',
    'Late (31-120 days)', 'Late (16-30 days)',
    'Does not meet the credit policy. Status:Charged Off',
    'Does not meet the credit policy. Status:Fully Paid'
])]
 
loans['target'] = np.where(
    loans['loan_status'].isin(['Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid']), 0, 1)
 
print("\nTarget distribution:")
print(loans['target'].value_counts())
print(f"Default rate: {loans['target'].mean():.2%}")
 
loans.drop('loan_status', axis=1, inplace=True)

# --- Step 3: Handle missing values ---
print("\nMissing values before cleaning:")
print(loans.isnull().sum())

# Fill numeric missing values with median
numeric_cols = loans.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    loans[col].fillna(loans[col].median(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = loans.select_dtypes(include=['object']).columns
for col in categorical_cols:
    loans[col].fillna(loans[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(loans.isnull().sum())

# --- Step 4: Clean emp_length ---
# Check if emp_length is text or already numeric
if loans['emp_length'].dtype == object:
    loans['emp_length'] = loans['emp_length'].str.replace(' years', '')
    loans['emp_length'] = loans['emp_length'].str.replace(' year', '')
    loans['emp_length'] = loans['emp_length'].str.replace('< 1', '0')
    loans['emp_length'] = loans['emp_length'].str.replace('10+', '10')
    loans['emp_length'] = pd.to_numeric(loans['emp_length'], errors='coerce')
    loans['emp_length'].fillna(loans['emp_length'].median(), inplace=True)
else:
    print("emp_length is already numeric — skipping string cleaning")

# --- Step 5: Clean int_rate ---
if loans['int_rate'].dtype == object:
    loans['int_rate'] = loans['int_rate'].str.replace('%', '').astype(float)

# --- Step 6: Clean revol_util ---
if loans['revol_util'].dtype == object:
    loans['revol_util'] = loans['revol_util'].str.replace('%', '').astype(float)
    loans['revol_util'].fillna(loans['revol_util'].median(), inplace=True)

print("\nData types after cleaning:")
print(loans.dtypes)
 
print("\nData types after cleaning:")
print(loans.dtypes) 
def calculate_woe_iv(df, feature, target, bins=10):   
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV)
    for a given feature against the target variable.
 
    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = Sum((Distribution of Events - Distribution of Non-Events) * WoE)               
 
    IV Interpretation:
    < 0.02  : Useless                                                                    
    0.02-0.1: Weak
    0.1-0.3 : Medium
    0.3-0.5 : Strong
    > 0.5   : Suspicious (too good)
    """
                                                                          
    df = df[[feature, target]].copy()
 
    # Bin numeric features
    if df[feature].dtype in [np.float64, np.int64]:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        df['bin'] = df[feature]
 
    grouped = df.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']
 
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
 
    grouped['dist_events'] = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events
 
    # Avoid log(0)
    grouped['dist_events'] = grouped['dist_events'].replace(0, 0.0001)               
    grouped['dist_non_events'] = grouped['dist_non_events'].replace(0, 0.0001)
 
    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['iv'] = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['woe']
 
    iv = grouped['iv'].sum()
    return grouped['woe'], iv
 
# Calculate IV for all features
print("\n--- Information Value for each feature ---")
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
print(iv_df)
 
# Keep features with IV > 0.02 (at least weak predictive power)
selected_features = iv_df[iv_df['IV'] > 0.02].index.tolist()
print(f"\nSelected features (IV > 0.02): {selected_features}")
 
# Plot IV values
plt.figure(figsize=(10, 6))    
iv_df[iv_df['IV'] > 0.02]['IV'].sort_values().plot(kind='barh', color='steelblue')
plt.title('Information Value by Feature', fontsize=14)
plt.xlabel('Information Value')
plt.tight_layout()
plt.savefig('iv_chart.png', dpi=150)
plt.show()
# PHASE 4 — LOGISTIC REGRESSION (PD MODEL)
# =============================================================
 
# --- WoE Encoding for selected features ---
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
loans_woe.dropna(inplace=True)
 
print("\nWoE encoded dataset shape:", loans_woe.shape)
 
# --- Train / Test Split ---
X = loans_woe.drop('target', axis=1)
y = loans_woe['target']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
 
print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
 
# --- Fit Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)  
lr.fit(X_train, y_train)
 
# --- Predictions ---
y_pred_prob = lr.predict_proba(X_test)[:, 1]
y_pred = lr.predict(X_test)
 
# --- Evaluation ---
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC Score: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

##Logistic regression with a different output, more statistical

X_train_sm = sm.add_constant(X_train)  # adds intercept
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
print(result.summary())


 
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
 
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — PD Model')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
 
# =============================================================
# PHASE 5 — SCORECARD SCALING
# =============================================================
# Convert log-odds (PD) to a credit score on a 300-850 scale
# Standard formula used by real banks:
# Score = Offset + Factor * log-odds
# Where: Factor = pdo / ln(2), Offset = base_score - Factor * ln(base_odds)
 
pdo = 20          # Points to double the odds
base_score = 600  # Score at base odds
base_odds = 1/19  # 1 bad for every 19 good (roughly 5% default rate)
 
factor = pdo / np.log(2)
offset = base_score - factor * np.log(base_odds)
 
# Get log-odds from model
log_odds = np.log(y_pred_prob / (1 - y_pred_prob + 1e-10))
scores = offset + factor * log_odds
 
# Clip to realistic range
scores = np.clip(scores, 300, 850)
 
print(f"\n--- Credit Score Distribution ---")
print(f"Min score:  {scores.min():.0f}")
print(f"Max score:  {scores.max():.0f}")
print(f"Mean score: {scores.mean():.0f}")
 
# Plot score distribution
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Credit Score Distribution', fontsize=14)
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=150)
plt.show()

##I am observing from the plot that the credit score might follow normal distribution. I'm going to run a Kolmogorov-Smirnoff test to check my hypothesis.
mean_score = scores.mean()
std_score = scores.std()
ks_stat, ks_p = stats.kstest(scores, 'norm', args=(mean_score, std_score))
print(f"\nKolmogorov-Smirnov Test on Score Distribution:")
print(f"KS Statistic: {ks_stat:.4f}")
print(f"P-value: {ks_p:.4f}")
if ks_p > 0.05:
    print("Scores follow a normal distribution (fail to reject H0)")
else:
    print("Scores do not follow a normal distribution (reject H0)")
 
# Score by actual default status
scores_df = pd.DataFrame({'score': scores, 'default': y_test.values})
print("\nAverage score by default status:")
print(scores_df.groupby('default')['score'].mean())
 
# =============================================================
# PHASE 6 — PSI (POPULATION STABILITY INDEX)
# =============================================================
# PSI measures how much the distribution of a variable has shifted
# between two time periods (e.g. training vs scoring population)
# PSI < 0.1  : No significant change
# PSI 0.1-0.2: Moderate change — monitor
# PSI > 0.2  : Significant shift — model may need retraining
 
def calculate_psi(expected, actual, bins=10):
    """
    Calculate PSI between expected (train) and actual (test) distributions.
    expected = array of predicted probabilities from training set
    actual   = array of predicted probabilities from test set
    """
    breakpoints = np.linspace(0, 1, bins + 1)
 
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
 
    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
 
    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
 
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = psi_values.sum()
    return psi, psi_values
 
# Compare PD distribution: train vs test
train_probs = lr.predict_proba(X_train)[:, 1]
test_probs = lr.predict_proba(X_test)[:, 1]
 
psi_score, psi_bins = calculate_psi(train_probs, test_probs)
 
print(f"\n--- PSI Result ---")
print(f"PSI = {psi_score:.4f}")
if psi_score < 0.1:
    print("Interpretation: No significant population shift — model is stable")
elif psi_score < 0.2:
    print("Interpretation: Moderate shift — monitor the model closely")
else:
    print("Interpretation: Significant shift — consider retraining the model")
 
# Plot PSI bins
plt.figure(figsize=(10, 5))
bins_labels = [f"{i*10}-{(i+1)*10}%" for i in range(len(psi_bins))]
plt.bar(bins_labels, psi_bins, color='steelblue', edgecolor='black', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.8)
plt.title(f'PSI by PD Bucket (Total PSI = {psi_score:.4f})', fontsize=13)
plt.xlabel('PD Bucket')
plt.ylabel('PSI Contribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('psi_chart.png', dpi=150)
plt.show()

