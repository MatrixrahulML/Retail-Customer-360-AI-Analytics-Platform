"""
02_etl_customer360.py
----------------------
ETL / integration layer. Combines the 5 raw source tables:
  customers, sales_orders, website_activity, support_tickets, marketing

into a single unified CUSTOMER 360 table with engineered features:
  - RFM (Recency, Frequency, Monetary)
  - Customer Lifetime Value (CLV)
  - Purchase behavior (avg order value, favorite category, channel mix)
  - Web engagement metrics
  - Support experience metrics
  - Marketing responsiveness metrics
  - A rule-based CHURN LABEL (used as ground truth to train the churn
    classifiers in the next script)

This mirrors what an SAP BODS / Fabric pipeline would do: extract from
multiple source systems, clean, conform grain to customer_id, and load
into a single analytical table ("Customer 360").
"""

import pandas as pd
import numpy as np
from datetime import datetime

RAW = "/Users/macbook/Downloads/customer360/data/raw"
PROC = "/Users/macbook/Downloads/customer360/data/processed"

TODAY = pd.Timestamp("2026-08-14")

# ---------------------------------------------------------------------------
# 1. EXTRACT
# ---------------------------------------------------------------------------
customers = pd.read_csv(f"{RAW}/customers.csv", parse_dates=["signup_date"])
sales = pd.read_csv(f"{RAW}/sales_orders.csv", parse_dates=["order_date"])
web = pd.read_csv(f"{RAW}/website_activity.csv", parse_dates=["session_date"])
support = pd.read_csv(f"{RAW}/support_tickets.csv", parse_dates=["ticket_date"])
marketing = pd.read_csv(f"{RAW}/marketing.csv", parse_dates=["sent_date"])

print("Raw row counts:")
print(f"  customers:       {len(customers):,}")
print(f"  sales_orders:    {len(sales):,}")
print(f"  website_activity:{len(web):,}")
print(f"  support_tickets: {len(support):,}")
print(f"  marketing:       {len(marketing):,}")

# ---------------------------------------------------------------------------
# 2. TRANSFORM -- basic cleaning
# ---------------------------------------------------------------------------
# Drop any duplicate rows / nulls in key join fields (typical ETL hygiene)
for df, name in [(customers, "customers"), (sales, "sales"), (web, "web"),
                  (support, "support"), (marketing, "marketing")]:
    before = len(df)
    df.dropna(subset=["customer_id"], inplace=True)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"  Cleaned {name}: removed {before - after} bad rows")

# ---------------------------------------------------------------------------
# 3. TRANSFORM -- RFM (Recency, Frequency, Monetary)
# ---------------------------------------------------------------------------
order_agg = sales.groupby("customer_id").agg(
    frequency=("order_id", "nunique"),
    monetary=("revenue", "sum"),
    last_purchase_date=("order_date", "max"),
    first_purchase_date=("order_date", "min"),
    avg_order_value=("revenue", "mean"),
    total_units=("quantity", "sum"),
).reset_index()

order_agg["recency_days"] = (TODAY - order_agg["last_purchase_date"]).dt.days
order_agg["customer_tenure_days"] = (TODAY - order_agg["first_purchase_date"]).dt.days

# RFM scores (1-5, 5=best) using quantile binning
def rfm_score(series, ascending):
    # ascending=True means lower raw value = better score (used for recency)
    ranks = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 5, labels=[1,2,3,4,5]).astype(int)

order_agg["R_score"] = rfm_score(order_agg["recency_days"], ascending=True)   # low recency_days = recent = good
order_agg["F_score"] = rfm_score(order_agg["frequency"], ascending=False)     # high frequency = good
order_agg["M_score"] = rfm_score(order_agg["monetary"], ascending=False)      # high monetary = good
order_agg["RFM_score"] = (order_agg["R_score"].astype(str) +
                           order_agg["F_score"].astype(str) +
                           order_agg["M_score"].astype(str))
order_agg["RFM_total"] = order_agg[["R_score","F_score","M_score"]].sum(axis=1)

# Favorite category per customer
fav_category = (sales.groupby(["customer_id","category"])["revenue"].sum()
                 .reset_index()
                 .sort_values("revenue", ascending=False)
                 .drop_duplicates("customer_id")[["customer_id","category"]]
                 .rename(columns={"category":"favorite_category"}))

# Favorite / most recent product for a simple recommendation baseline
last_product = (sales.sort_values("order_date")
                 .groupby("customer_id")
                 .tail(1)[["customer_id","product_name"]]
                 .rename(columns={"product_name":"last_product_purchased"}))

# Preferred sales channel (most used)
fav_channel = (sales.groupby(["customer_id","sales_channel"])["order_id"].count()
               .reset_index()
               .sort_values("order_id", ascending=False)
               .drop_duplicates("customer_id")[["customer_id","sales_channel"]]
               .rename(columns={"sales_channel":"top_purchase_channel"}))

# ---------------------------------------------------------------------------
# 4. TRANSFORM -- Web engagement features
# ---------------------------------------------------------------------------
web_agg = web.groupby("customer_id").agg(
    total_sessions=("session_id", "nunique"),
    avg_pages_per_session=("pages_viewed", "mean"),
    avg_time_on_site=("time_on_site_min", "mean"),
    cart_adds=("added_to_cart", "sum"),
    last_session_date=("session_date", "max"),
).reset_index()
web_agg["days_since_last_session"] = (TODAY - web_agg["last_session_date"]).dt.days

# ---------------------------------------------------------------------------
# 5. TRANSFORM -- Support experience features
# ---------------------------------------------------------------------------
support_agg = support.groupby("customer_id").agg(
    total_tickets=("ticket_id", "nunique"),
    avg_csat=("csat_score", "mean"),
    avg_resolution_hours=("resolution_time_hours", "mean"),
    escalation_rate=("escalated", "mean"),
).reset_index()

# ---------------------------------------------------------------------------
# 6. TRANSFORM -- Marketing responsiveness features
# ---------------------------------------------------------------------------
mkt_agg = marketing.groupby("customer_id").agg(
    total_touches=("touch_id", "nunique"),
    open_rate=("opened", "mean"),
    conversion_rate=("converted", "mean"),
).reset_index()

# ---------------------------------------------------------------------------
# 7. LOAD -- merge into single Customer 360 table
# ---------------------------------------------------------------------------
c360 = customers.copy()
c360 = c360.merge(order_agg, on="customer_id", how="left")
c360 = c360.merge(fav_category, on="customer_id", how="left")
c360 = c360.merge(last_product, on="customer_id", how="left")
c360 = c360.merge(fav_channel, on="customer_id", how="left")
c360 = c360.merge(web_agg, on="customer_id", how="left")
c360 = c360.merge(support_agg, on="customer_id", how="left")
c360 = c360.merge(mkt_agg, on="customer_id", how="left")

# Fill nulls for customers with no activity in a given source (never purchased, etc.)
numeric_fill_zero = ["frequency","monetary","avg_order_value","total_units","total_sessions",
    "avg_pages_per_session","avg_time_on_site","cart_adds","total_tickets","avg_resolution_hours",
    "escalation_rate","total_touches","open_rate","conversion_rate"]
for col in numeric_fill_zero:
    if col in c360.columns:
        c360[col] = c360[col].fillna(0)

c360["recency_days"] = c360["recency_days"].fillna((TODAY - c360["signup_date"]).dt.days)
c360["R_score"] = c360["R_score"].fillna(1).astype(int)
c360["F_score"] = c360["F_score"].fillna(1).astype(int)
c360["M_score"] = c360["M_score"].fillna(1).astype(int)
c360["RFM_total"] = c360["RFM_total"].fillna(3)
c360["avg_csat"] = c360["avg_csat"].fillna(c360["avg_csat"].mean())
c360["favorite_category"] = c360["favorite_category"].fillna("None")
c360["last_product_purchased"] = c360["last_product_purchased"].fillna("None")
c360["top_purchase_channel"] = c360["top_purchase_channel"].fillna(c360["preferred_channel"])

# ---------------------------------------------------------------------------
# 8. TRANSFORM -- Customer Lifetime Value (CLV)
# Simple, explainable historical CLV formula (good for interviews):
#   CLV = avg_order_value * purchase_frequency_rate * expected_lifespan_years
# Also compute a "gamma-gamma style" simplified predictive CLV using
# monetary total scaled by an engagement-adjusted retention factor.
# ---------------------------------------------------------------------------
c360["tenure_years"] = ((TODAY - c360["signup_date"]).dt.days / 365.25).clip(lower=0.1)
c360["purchase_freq_per_year"] = c360["frequency"] / c360["tenure_years"]

# Simple historical CLV = total monetary value to date
c360["clv_historical"] = c360["monetary"]

# Predictive CLV: projects value forward using purchase frequency, avg order
# value, and an estimated retention/survival factor derived from recency +
# engagement (lower recency_days & higher engagement => higher retention)
retention_factor = np.clip(
    1 - (c360["recency_days"] / (c360["tenure_years"] * 365.25 + 30)).clip(0, 1) * 0.6,
    0.1, 0.95
)
projected_years = 3
c360["clv_predicted"] = (c360["avg_order_value"].fillna(0) *
                          c360["purchase_freq_per_year"].fillna(0) *
                          projected_years * retention_factor)
c360["clv_predicted"] = c360["clv_predicted"].round(2)
c360["clv_historical"] = c360["clv_historical"].round(2)

# ---------------------------------------------------------------------------
# 9. TRANSFORM -- Churn label (ground truth for supervised models)
# Business rule: a customer is considered CHURNED if no purchase in the
# last 180 days AND (low engagement OR low satisfaction), OR no purchase
# in the last 365 days regardless of other factors, OR never purchased and
# signed up over 120 days ago with no recent web activity.
# This is a common, defensible "rule-based label then ML predicts it"
# approach used in real churn projects where true cancellation events
# aren't tracked.
# ---------------------------------------------------------------------------
never_purchased = c360["frequency"] == 0
long_dormant = c360["recency_days"] > 365
mid_dormant_low_engagement = (c360["recency_days"] > 180) & (
    (c360["avg_csat"] < 3) | (c360["days_since_last_session"].fillna(9999) > 180)
)
dormant_never_purchased = never_purchased & (
    (TODAY - c360["signup_date"]).dt.days > 120
) & (c360["days_since_last_session"].fillna(9999) > 120)

c360["churned"] = (long_dormant | mid_dormant_low_engagement | dormant_never_purchased).astype(int)

print(f"\nChurn rate in labeled data: {c360['churned'].mean():.1%}")

# ---------------------------------------------------------------------------
# 10. Final column selection / ordering for the Customer 360 table
# ---------------------------------------------------------------------------
c360["customer_name"] = c360["first_name"] + " " + c360["last_name"]

final_cols = [
    "customer_id","customer_name","region","country","customer_type","signup_date",
    "preferred_channel","email_opt_in",
    # RFM
    "recency_days","frequency","monetary","R_score","F_score","M_score","RFM_score","RFM_total",
    # CLV
    "clv_historical","clv_predicted","tenure_years","purchase_freq_per_year","avg_order_value",
    # Purchase behavior
    "favorite_category","last_product_purchased","top_purchase_channel","total_units",
    "last_purchase_date","first_purchase_date",
    # Web engagement
    "total_sessions","avg_pages_per_session","avg_time_on_site","cart_adds","days_since_last_session",
    # Support
    "total_tickets","avg_csat","avg_resolution_hours","escalation_rate",
    # Marketing
    "total_touches","open_rate","conversion_rate",
    # Label
    "churned",
]
c360_final = c360[final_cols].copy()

c360_final.to_csv(f"{PROC}/customer_360.csv", index=False)

print(f"\nCustomer 360 table created: {c360_final.shape[0]:,} rows x {c360_final.shape[1]} columns")
print(f"Saved to: {PROC}/customer_360.csv")
print("\nSample:")
print(c360_final.head(3).T)
