"""
04_churn_models.py
--------------------
Churn prediction using:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (non-linear, feature importance)

Target: `churned` (rule-based label created in the ETL step)

Outputs:
  - models/logreg_model.pkl, models/rf_model.pkl, models/churn_scaler.pkl
  - output/model_comparison.csv
  - output/rf_feature_importance.png
  - output/confusion_matrices.png
  - customer_360.csv updated with churn_probability (from best model) + predicted_churn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib

PROC = "/Users/macbook/Downloads/customer360/data/processed"
OUT = "/Users/macbook/Downloads/customer360/output"
MODELS = "/Users/macbook/Downloads/customer360/models"

df = pd.read_csv(f"{PROC}/customer_360.csv")

# ---------------------------------------------------------------------------
# Feature selection
# Note: recency_days / days_since_last_session are intentionally EXCLUDED
# from the model or handled carefully -- since the churn label rule is
# partly defined using recency, including it directly would cause label
# leakage that inflates accuracy unrealistically. We keep frequency,
# monetary, engagement, satisfaction, and marketing features which are
# genuinely predictive without being a restatement of the label.
# For a realistic-but-still-strong result, we keep a *capped* recency
# feature but clearly flag this choice for transparency in interviews.
# ---------------------------------------------------------------------------
feature_cols = [
    "frequency", "monetary", "avg_order_value", "purchase_freq_per_year",
    "total_sessions", "avg_pages_per_session", "avg_time_on_site", "cart_adds",
    "total_tickets", "avg_csat", "avg_resolution_hours", "escalation_rate",
    "total_touches", "open_rate", "conversion_rate",
    "tenure_years", "email_opt_in", "RFM_total",
]

X = df[feature_cols].fillna(0)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Model 1: Logistic Regression
# ---------------------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# Model 2: Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)  # tree models don't need scaling
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }

results = [
    evaluate("Logistic Regression", y_test, logreg_pred, logreg_proba),
    evaluate("Random Forest", y_test, rf_pred, rf_proba),
]
results_df = pd.DataFrame(results)
print("Model comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, pred) in zip(axes, [("Logistic Regression", logreg_pred), ("Random Forest", rf_pred)]):
    cm = confusion_matrix(y_test, pred)
    im = ax.imshow(cm, cmap="Reds")
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Retained","Churned"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Retained","Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(6,5))
for name, proba in [("Logistic Regression", logreg_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Models")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Random Forest feature importance
# ---------------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(8,6))
importances.plot(kind="barh", color="#c0392b")
plt.title("Random Forest - Feature Importance (Churn Prediction)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/rf_feature_importance.png", dpi=150)
plt.close()

print("\nTop 5 churn drivers (Random Forest):")
print(importances.sort_values(ascending=False).head(5))

# ---------------------------------------------------------------------------
# Save models
# ---------------------------------------------------------------------------
joblib.dump(logreg, f"{MODELS}/logreg_model.pkl")
joblib.dump(rf, f"{MODELS}/rf_model.pkl")
joblib.dump(scaler, f"{MODELS}/churn_scaler.pkl")
joblib.dump(feature_cols, f"{MODELS}/churn_feature_cols.pkl")

# ---------------------------------------------------------------------------
# Score ALL customers with best model (Random Forest, typically stronger)
# and write churn_probability + predicted_churn back into customer_360
# ---------------------------------------------------------------------------
best_model = rf if results_df.loc[results_df["roc_auc"].idxmax(), "model"] == "Random Forest" else logreg
X_all = df[feature_cols].fillna(0)
if best_model is rf:
    df["churn_probability"] = rf.predict_proba(X_all)[:, 1].round(4)
else:
    df["churn_probability"] = logreg.predict_proba(scaler.transform(X_all))[:, 1].round(4)

df["predicted_churn"] = (df["churn_probability"] >= 0.5).astype(int)

df.to_csv(f"{PROC}/customer_360.csv", index=False)
print(f"\nBest model by ROC-AUC: {results_df.loc[results_df['roc_auc'].idxmax(), 'model']}")
print("customer_360.csv updated with churn_probability + predicted_churn")
