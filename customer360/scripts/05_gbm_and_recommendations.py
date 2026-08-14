"""
05_gbm_and_recommendations.py
-------------------------------
SANDBOX-RUNNABLE VERSION of script 05. This environment has no internet
access to `pip install xgboost`, so this version uses
sklearn.ensemble.GradientBoostingClassifier -- the same gradient-boosted
trees algorithm family as XGBoost, just the (slower, no-GPU) reference
implementation that ships with scikit-learn.

Everything else is identical in approach to 05_xgboost_and_recommendations.py.
When you run that script locally with `pip install xgboost`, you should get
very similar or slightly better metrics from XGBoost's more efficient
regularized boosting.

Outputs:
  - models/gbm_model.pkl
  - output/model_comparison.csv (updated with Gradient Boosting row)
  - output/gbm_feature_importance.png
  - customer_360.csv updated with churn_probability (from GBM) + recommended_product
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
import joblib

PROC = "/Users/macbook/Downloads/customer360/data/processed"
RAW = "/Users/macbook/Downloads/customer360/data/raw"
OUT = "/Users/macbook/Downloads/customer360/output"
MODELS = "/Users/macbook/Downloads/customer360/models"

df = pd.read_csv(f"{PROC}/customer_360.csv")
sales = pd.read_csv(f"{RAW}/sales_orders.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------------------
# PART 1: Gradient Boosting churn model (stand-in for XGBoost)
# ---------------------------------------------------------------------------
feature_cols = joblib.load(f"{MODELS}/churn_feature_cols.pkl")
X = df[feature_cols].fillna(0)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# GradientBoostingClassifier has no built-in class_weight, so we
# oversample the minority (churned) class in the training set to
# approximate what scale_pos_weight does in XGBoost
train = X_train.copy()
train["churned"] = y_train.values
majority = train[train["churned"] == 0]
minority = train[train["churned"] == 1]
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
train_balanced = pd.concat([majority, minority_upsampled])
X_train_bal = train_balanced[feature_cols]
y_train_bal = train_balanced["churned"]

gbm = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
gbm.fit(X_train_bal, y_train_bal)
gbm_pred = gbm.predict(X_test)
gbm_proba = gbm.predict_proba(X_test)[:, 1]

gbm_results = {
    "model": "Gradient Boosting (XGBoost stand-in)",
    "accuracy": round(accuracy_score(y_test, gbm_pred), 4),
    "precision": round(precision_score(y_test, gbm_pred), 4),
    "recall": round(recall_score(y_test, gbm_pred), 4),
    "f1_score": round(f1_score(y_test, gbm_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, gbm_proba), 4),
}
print("Gradient Boosting results:", gbm_results)

comparison_path = f"{OUT}/model_comparison.csv"
try:
    existing = pd.read_csv(comparison_path)
    existing = existing[~existing["model"].isin(["Gradient Boosting (XGBoost stand-in)", "XGBoost"])]
    combined = pd.concat([existing, pd.DataFrame([gbm_results])], ignore_index=True)
except FileNotFoundError:
    combined = pd.DataFrame([gbm_results])
combined.to_csv(comparison_path, index=False)
print(combined)

importances = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(8,6))
importances.plot(kind="barh", color="#2c3e50")
plt.title("Gradient Boosting - Feature Importance (Churn Prediction)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/gbm_feature_importance.png", dpi=150)
plt.close()

joblib.dump(gbm, f"{MODELS}/gbm_model.pkl")

X_all = df[feature_cols].fillna(0)
df["churn_probability"] = gbm.predict_proba(X_all)[:, 1].round(4)
df["predicted_churn"] = (df["churn_probability"] >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# PART 2: Product recommendation engine (same design as 05_xgboost version,
# using GradientBoostingClassifier as the re-ranker)
# ---------------------------------------------------------------------------
prod_df = sales[["product_id","product_name","category","unit_price"]].drop_duplicates("product_id")

cust_category_pref = (sales.groupby(["customer_id","category"])["revenue"].sum()
                       .unstack(fill_value=0))
cust_category_pref = cust_category_pref.div(cust_category_pref.sum(axis=1).replace(0,1), axis=0)

rng = np.random.default_rng(42)
train_rows = []
customers_with_orders = sales["customer_id"].unique()

for cid in customers_with_orders:
    bought = set(sales.loc[sales["customer_id"] == cid, "product_id"])
    not_bought = [p for p in prod_df["product_id"] if p not in bought]
    neg_sample = rng.choice(not_bought, size=min(len(bought), len(not_bought)), replace=False) if not_bought else []
    for pid in bought:
        train_rows.append((cid, pid, 1))
    for pid in neg_sample:
        train_rows.append((cid, pid, 0))

reco_train = pd.DataFrame(train_rows, columns=["customer_id","product_id","label"])
reco_train = reco_train.merge(prod_df, on="product_id", how="left")

# vectorized category affinity lookup (faster than row-wise apply used in the xgboost script)
cust_cat_long = cust_category_pref.stack().rename("category_affinity").reset_index()
cust_cat_long.columns = ["customer_id", "category", "category_affinity"]
reco_train = reco_train.merge(cust_cat_long, on=["customer_id","category"], how="left")
reco_train["category_affinity"] = reco_train["category_affinity"].fillna(0.0)

reco_train = reco_train.merge(
    df[["customer_id","avg_order_value","frequency","RFM_total"]], on="customer_id", how="left"
)
reco_train["price_diff_from_avg_order"] = (reco_train["unit_price"] - reco_train["avg_order_value"]).abs()

reco_features = ["category_affinity","avg_order_value","frequency","RFM_total",
                  "unit_price","price_diff_from_avg_order"]
Xr = reco_train[reco_features].fillna(0)
yr = reco_train["label"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
reco_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
reco_model.fit(Xr_train, yr_train)
reco_auc = roc_auc_score(yr_test, reco_model.predict_proba(Xr_test)[:,1])
print(f"\nRecommendation model ROC-AUC: {reco_auc:.4f}")
joblib.dump(reco_model, f"{MODELS}/recommendation_model.pkl")

# Vectorized scoring: build full customer x candidate-product cross join
# per customer is expensive at 5000 customers x 20 products = 100k rows,
# which is fine to do in one vectorized pass instead of a python loop.
all_customers = df[["customer_id","avg_order_value","frequency","RFM_total"]].copy()
all_customers["key"] = 1
prod_df2 = prod_df.copy()
prod_df2["key"] = 1
cross = all_customers.merge(prod_df2, on="key").drop(columns="key")

# remove already-purchased combos
purchased_pairs = set(zip(sales["customer_id"], sales["product_id"]))
cross["already_bought"] = cross.apply(lambda r: (r["customer_id"], r["product_id"]) in purchased_pairs, axis=1)
cross = cross[~cross["already_bought"]].drop(columns="already_bought")

cross = cross.merge(cust_cat_long, on=["customer_id","category"], how="left")
cross["category_affinity"] = cross["category_affinity"].fillna(0.0)
cross["price_diff_from_avg_order"] = (cross["unit_price"] - cross["avg_order_value"]).abs()

cross["score"] = reco_model.predict_proba(cross[reco_features].fillna(0))[:,1]
best_reco = (cross.sort_values("score", ascending=False)
             .drop_duplicates("customer_id")[["customer_id","product_name","score"]]
             .rename(columns={"product_name":"recommended_product","score":"recommendation_score"}))

df = df.merge(best_reco, on="customer_id", how="left")
df["recommended_product"] = df["recommended_product"].fillna("None")
df["recommendation_score"] = df["recommendation_score"].fillna(0.0).round(4)

df.to_csv(f"{PROC}/customer_360.csv", index=False)
print("\ncustomer_360.csv updated with churn_probability (GBM) + recommended_product")
print(df[["customer_id","churn_probability","recommended_product","recommendation_score"]].head(10))
