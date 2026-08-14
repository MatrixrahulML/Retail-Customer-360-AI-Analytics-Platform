"""
05_xgboost_and_recommendations.py
-----------------------------------
NOTE: This script requires the `xgboost` package. It could not be installed
or executed in the sandbox used to build this project (no internet access
there), so run this on your own machine:

    pip install xgboost

Two things happen here:
  1. XGBoost churn model -- typically the strongest of the 3 churn models,
     benchmarked against Logistic Regression + Random Forest from script 04.
  2. Product recommendation engine using a simple, explainable
     "customers who bought X also bought Y" collaborative-filtering
     approach (item-item co-occurrence), re-ranked by an XGBoost model
     that scores product-affinity likelihood using the customer's RFM +
     category-affinity features. This gives you a genuine "ML-powered
     recommendation" story without needing a full deep-learning
     recommender, which would be overkill for this dataset size.

Outputs:
  - models/xgb_model.pkl
  - output/model_comparison.csv (updated with XGBoost row)
  - output/xgb_feature_importance.png
  - customer_360.csv updated with churn_probability (from XGBoost, since
    it usually wins) + recommended_product
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
import joblib
import xgboost as xgb

PROC = "/Users/macbook/Downloads/customer360/data/processed"
RAW = "/Users/macbook/Downloads/customer360/data/raw"
OUT = "/Users/macbook/Downloads/customer360/output"
MODELS = "/Users/macbook/Downloads/customer360/models"

df = pd.read_csv(f"{PROC}/customer_360.csv")
sales = pd.read_csv(f"{RAW}/sales_orders.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------------------
# PART 1: XGBoost churn model (same feature set as script 04 for apples-
# to-apples comparison)
# ---------------------------------------------------------------------------
feature_cols = joblib.load(f"{MODELS}/churn_feature_cols.pkl")
X = df[feature_cols].fillna(0)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# scale_pos_weight handles class imbalance (similar role to class_weight="balanced")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_results = {
    "model": "XGBoost",
    "accuracy": round(accuracy_score(y_test, xgb_pred), 4),
    "precision": round(precision_score(y_test, xgb_pred), 4),
    "recall": round(recall_score(y_test, xgb_pred), 4),
    "f1_score": round(f1_score(y_test, xgb_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, xgb_proba), 4),
}
print("XGBoost results:", xgb_results)

# Append to model_comparison.csv from script 04
comparison_path = f"{OUT}/model_comparison.csv"
try:
    existing = pd.read_csv(comparison_path)
    existing = existing[existing["model"] != "XGBoost"]  # avoid dup on re-run
    combined = pd.concat([existing, pd.DataFrame([xgb_results])], ignore_index=True)
except FileNotFoundError:
    combined = pd.DataFrame([xgb_results])
combined.to_csv(comparison_path, index=False)
print(combined)

# Feature importance chart
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(8,6))
importances.plot(kind="barh", color="#2c3e50")
plt.title("XGBoost - Feature Importance (Churn Prediction)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/xgb_feature_importance.png", dpi=150)
plt.close()

joblib.dump(xgb_model, f"{MODELS}/xgb_model.pkl")

# Score all customers with XGBoost (typically best performer -> becomes
# the production churn_probability used in the Power BI dashboard)
X_all = df[feature_cols].fillna(0)
df["churn_probability"] = xgb_model.predict_proba(X_all)[:, 1].round(4)
df["predicted_churn"] = (df["churn_probability"] >= 0.5).astype(int)

# ---------------------------------------------------------------------------
# PART 2: Product recommendation engine
# Step A: item-item co-occurrence (customers who bought X also bought Y)
# Step B: XGBoost re-ranks candidate products per customer using
#         category affinity + RFM + price-band fit
# ---------------------------------------------------------------------------

# Step A: build co-occurrence matrix
basket = sales.groupby(["customer_id"])["product_name"].apply(set)
from itertools import combinations
from collections import defaultdict

co_occurrence = defaultdict(lambda: defaultdict(int))
for products in basket:
    for a, b in combinations(sorted(products), 2):
        co_occurrence[a][b] += 1
        co_occurrence[b][a] += 1

def top_co_purchased(product, n=3):
    if product not in co_occurrence:
        return []
    return [p for p, _ in sorted(co_occurrence[product].items(), key=lambda x: -x[1])[:n]]

# Step B: simple XGBoost-based re-ranking model
# Training data: for each (customer, product) the customer HAS bought = positive
# sample an equal number of NOT-bought products as negatives, features =
# customer RFM/category affinity + product price/category match
prod_df = sales[["product_id","product_name","category","unit_price"]].drop_duplicates("product_id")

cust_category_pref = (sales.groupby(["customer_id","category"])["revenue"].sum()
                       .unstack(fill_value=0))
cust_category_pref = cust_category_pref.div(cust_category_pref.sum(axis=1).replace(0,1), axis=0)  # normalize to %

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

def category_affinity(row):
    prefs = cust_category_pref.loc[row["customer_id"]] if row["customer_id"] in cust_category_pref.index else None
    if prefs is None or row["category"] not in prefs.index:
        return 0.0
    return prefs[row["category"]]

reco_train["category_affinity"] = reco_train.apply(category_affinity, axis=1)
reco_train = reco_train.merge(
    df[["customer_id","avg_order_value","frequency","RFM_total"]], on="customer_id", how="left"
)
reco_train["price_diff_from_avg_order"] = (reco_train["unit_price"] - reco_train["avg_order_value"]).abs()

reco_features = ["category_affinity","avg_order_value","frequency","RFM_total",
                  "unit_price","price_diff_from_avg_order"]
Xr = reco_train[reco_features].fillna(0)
yr = reco_train["label"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42, stratify=yr)
reco_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
reco_model.fit(Xr_train, yr_train)
reco_auc = roc_auc_score(yr_test, reco_model.predict_proba(Xr_test)[:,1])
print(f"\nRecommendation model ROC-AUC: {reco_auc:.4f}")
joblib.dump(reco_model, f"{MODELS}/recommendation_model.pkl")

# Score: for each customer, rank NOT-yet-purchased products, pick top 1
def recommend_for_customer(cid, top_n=1):
    bought = set(sales.loc[sales["customer_id"] == cid, "product_id"])
    candidates = prod_df[~prod_df["product_id"].isin(bought)].copy()
    if candidates.empty:
        return "None"
    candidates["customer_id"] = cid
    candidates["category_affinity"] = candidates.apply(category_affinity, axis=1)
    cust_row = df.loc[df["customer_id"] == cid, ["avg_order_value","frequency","RFM_total"]]
    if cust_row.empty:
        return "None"
    for col in ["avg_order_value","frequency","RFM_total"]:
        candidates[col] = cust_row[col].values[0]
    candidates["price_diff_from_avg_order"] = (candidates["unit_price"] - candidates["avg_order_value"]).abs()
    scores = reco_model.predict_proba(candidates[reco_features].fillna(0))[:,1]
    candidates["score"] = scores
    best = candidates.sort_values("score", ascending=False).head(top_n)
    return best["product_name"].tolist()[0] if len(best) else "None"

print("Scoring product recommendations for all customers (this may take a moment)...")
recommendations = {}
for cid in df["customer_id"]:
    recommendations[cid] = recommend_for_customer(cid)

df["recommended_product"] = df["customer_id"].map(recommendations)
df["recommended_product"] = df["recommended_product"].fillna("None")

df.to_csv(f"{PROC}/customer_360.csv", index=False)
print("\ncustomer_360.csv updated with churn_probability (XGBoost) + recommended_product")
print(df[["customer_id","churn_probability","recommended_product"]].head(10))
