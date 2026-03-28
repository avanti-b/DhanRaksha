"""
AI Fraud Detection - Model Training Script
==========================================
This script does everything:
1. Downloads the dataset
2. Cleans and preprocesses the data
3. Handles class imbalance with SMOTE
4. Trains Logistic Regression and Random Forest
5. Evaluates both models
6. Saves the best model for the Flask API

HOW TO RUN:
  - On your computer:  python train_model.py
  - On Google Colab:   Upload this file and run it there (recommended for speed)

DATASET:
  Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  File name: creditcard.csv
  Place it in the same folder as this script.
"""

# ─────────────────────────────────────────────
# STEP 0: Install required libraries
# (Uncomment the line below if running on Google Colab)
# ─────────────────────────────────────────────
# !pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works on Colab and servers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = "creditcard.csv"       # Place the Kaggle CSV here
MODEL_DIR    = "model"                # Folder to save trained model
PLOTS_DIR    = "plots"                # Folder to save plots
RANDOM_SEED  = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("  FraudShield AI — Model Training Pipeline")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/6] Loading dataset...")

if not os.path.exists(DATASET_PATH):
    print(f"""
❌ Dataset not found at '{DATASET_PATH}'

👉 Download it from:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

👉 Then place 'creditcard.csv' in the same folder as this script.
""")
    exit(1)

df = pd.read_csv(DATASET_PATH)
print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")


# ─────────────────────────────────────────────
# STEP 2: DATA CLEANING & EXPLORATION
# ─────────────────────────────────────────────
print("\n[2/6] Cleaning and exploring data...")

# 2a. Check for missing values
missing = df.isnull().sum().sum()
print(f"   Missing values: {missing}")
if missing > 0:
    df.dropna(inplace=True)
    print(f"   Dropped rows with missing values. Remaining: {len(df):,}")

# 2b. Check for duplicates
dupes = df.duplicated().sum()
print(f"   Duplicate rows: {dupes}")
if dupes > 0:
    df.drop_duplicates(inplace=True)
    print(f"   Removed duplicates. Remaining: {len(df):,}")

# 2c. Class distribution
fraud_count   = df['Class'].sum()
legit_count   = len(df) - fraud_count
fraud_pct     = fraud_count / len(df) * 100
print(f"\n   Class distribution:")
print(f"   ✅ Legitimate : {legit_count:,} ({100-fraud_pct:.2f}%)")
print(f"   🚨 Fraudulent : {fraud_count:,}  ({fraud_pct:.4f}%)")

# 2d. Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Legitimate', 'Fraudulent'], [legit_count, fraud_count],
            color=['#3266ad', '#c0392b'], alpha=0.85)
axes[0].set_title('Class Distribution (Raw)')
axes[0].set_ylabel('Count')
for i, v in enumerate([legit_count, fraud_count]):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10)

# Amount distribution by class
df[df['Class'] == 0]['Amount'].hist(ax=axes[1], bins=50, alpha=0.6,
                                    label='Legitimate', color='#3266ad')
df[df['Class'] == 1]['Amount'].hist(ax=axes[1], bins=50, alpha=0.6,
                                    label='Fraudulent', color='#c0392b')
axes[1].set_title('Transaction Amount by Class')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/01_class_distribution.png")

# 2e. Feature correlation heatmap (V1-V10 for readability)
fig, ax = plt.subplots(figsize=(10, 6))
v_cols = [c for c in df.columns if c.startswith('V')][:14]
corr = df[v_cols + ['Amount', 'Class']].corr()
sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0,
            annot=False, linewidths=0.3, fmt=".1f")
ax.set_title('Feature Correlation Heatmap (V1–V14 + Amount)')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/02_correlation_heatmap.png")

# 2f. Fraud by hour of day
df['Hour'] = (df['Time'] // 3600) % 24
hourly_fraud = df[df['Class']==1].groupby('Hour').size()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly_fraud.index, hourly_fraud.values, color='#c0392b',
        marker='o', linewidth=2)
ax.fill_between(hourly_fraud.index, hourly_fraud.values, alpha=0.15, color='#c0392b')
ax.set_title('Fraudulent Transactions by Hour of Day')
ax.set_xlabel('Hour')
ax.set_ylabel('Fraud Count')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_fraud_by_hour.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/03_fraud_by_hour.png")


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n[3/6] Preprocessing features...")

# Separate features (X) and label (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Drop Hour column we added (it was for EDA only, Time encodes this already)
if 'Hour' in X.columns:
    X = X.drop(columns=['Hour'])

# Scale 'Time' and 'Amount' — V1-V28 are already PCA-scaled
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

print(f"   Feature matrix shape: {X.shape}")
print(f"   Scaled: Time, Amount")
print(f"   Features used: {list(X.columns)}")

# Train/test split (stratified to preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)
print(f"   Train size: {len(X_train):,} | Test size: {len(X_test):,}")
print(f"   Train fraud: {y_train.sum():,} | Test fraud: {y_test.sum():,}")


# ─────────────────────────────────────────────
# STEP 4: HANDLE CLASS IMBALANCE (SMOTE)
# ─────────────────────────────────────────────
print("\n[4/6] Applying SMOTE to handle class imbalance...")

print(f"   Before SMOTE — Legitimate: {(y_train==0).sum():,} | Fraud: {(y_train==1).sum():,}")

smote = SMOTE(random_state=RANDOM_SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"   After  SMOTE — Legitimate: {(y_train_sm==0).sum():,} | Fraud: {(y_train_sm==1).sum():,}")
print(f"   ✅ Dataset balanced for training.")

# Plot before/after SMOTE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Legitimate', 'Fraudulent'], [(y_train==0).sum(), (y_train==1).sum()],
            color=['#3266ad', '#c0392b'], alpha=0.85)
axes[0].set_title('Before SMOTE')
axes[0].set_ylabel('Count')
axes[1].bar(['Legitimate', 'Fraud (synthetic)'],
            [(y_train_sm==0).sum(), (y_train_sm==1).sum()],
            color=['#3266ad', '#27ae60'], alpha=0.85)
axes[1].set_title('After SMOTE (Balanced)')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_smote_comparison.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/04_smote_comparison.png")


# ─────────────────────────────────────────────
# STEP 5: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[5/6] Training models...")

def evaluate_model(name, clf, X_tr, y_tr, X_te, y_te):
    """Train, predict, and return metrics for a model."""
    print(f"\n   Training {name}...")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    prec  = precision_score(y_te, y_pred)
    rec   = recall_score(y_te, y_pred)
    f1    = f1_score(y_te, y_pred)
    auc   = roc_auc_score(y_te, y_prob)
    cm    = confusion_matrix(y_te, y_pred)

    print(f"   ✅ {name} Results:")
    print(f"      Precision : {prec:.4f}")
    print(f"      Recall    : {rec:.4f}")
    print(f"      F1 Score  : {f1:.4f}")
    print(f"      AUC-ROC   : {auc:.4f}")
    print(f"      Confusion Matrix:\n{cm}")

    return {
        "name": name,
        "model": clf,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "confusion_matrix": cm.tolist()
        }
    }

# 5a. Logistic Regression (baseline)
lr_result = evaluate_model(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED),
    X_train_sm, y_train_sm, X_test, y_test
)

# 5b. Random Forest (main model)
rf_result = evaluate_model(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1          # Use all CPU cores
    ),
    X_train_sm, y_train_sm, X_test, y_test
)


# ─────────────────────────────────────────────
# STEP 6: EVALUATE & SAVE
# ─────────────────────────────────────────────
print("\n[6/6] Saving plots and model...")

# 6a. Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, res in zip(axes, [lr_result, rf_result]):
    cm = np.array(res['metrics']['confusion_matrix'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Legitimate', 'Fraud'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(res['name'])
plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_confusion_matrices.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/05_confusion_matrices.png")

# 6b. ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for res, color, ls in [
    (lr_result, '#73726c', '--'),
    (rf_result, '#3266ad', '-'),
]:
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2, ls=ls,
            label=f"{res['name']} (AUC={res['metrics']['auc_roc']:.3f})")
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_roc_curves.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/06_roc_curves.png")

# 6c. Feature importance (Random Forest)
importances = rf_result['model'].feature_importances_
feat_names  = X.columns.tolist()
feat_df     = pd.DataFrame({'feature': feat_names, 'importance': importances})
feat_df     = feat_df.sort_values('importance', ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feat_df['feature'], feat_df['importance'], color='#3266ad', alpha=0.85)
ax.set_title('Top 15 Feature Importances (Random Forest)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=150)
plt.close()
print(f"   📊 Plot saved: {PLOTS_DIR}/07_feature_importance.png")

# 6d. Save the best model (Random Forest) and scaler
joblib.dump(rf_result['model'], f"{MODEL_DIR}/fraud_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
print(f"   💾 Model saved: {MODEL_DIR}/fraud_model.pkl")
print(f"   💾 Scaler saved: {MODEL_DIR}/scaler.pkl")

# 6e. Save metrics JSON (for the dashboard API)
all_metrics = {
    "random_forest": rf_result['metrics'],
    "logistic_regression": lr_result['metrics'],
    "dataset_info": {
        "total_transactions": int(len(df)),
        "fraud_count": int(fraud_count),
        "legitimate_count": int(legit_count),
        "fraud_percentage": round(fraud_pct, 4),
        "features": X.columns.tolist()
    }
}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"   💾 Metrics saved: {MODEL_DIR}/metrics.json")

# Final summary
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"""
  Random Forest  → F1: {rf_result['metrics']['f1_score']:.4f}  AUC: {rf_result['metrics']['auc_roc']:.4f}
  Logistic Reg   → F1: {lr_result['metrics']['f1_score']:.4f}  AUC: {lr_result['metrics']['auc_roc']:.4f}

  Best model (Random Forest) saved to: model/fraud_model.pkl
  All plots saved to: plots/

  Next step → Run the Flask backend:
  $ python app.py
""")
