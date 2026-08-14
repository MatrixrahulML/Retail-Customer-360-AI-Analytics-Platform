"""
06_export_for_powerbi.py
--------------------------
Prepares final, clean, Power BI-ready CSV exports.

Outputs (in /output):
  - customer_360_dashboard.csv  -- main fact table for Power BI, with the
    exact fields the dashboard needs: Customer Value, Churn Probability,
    Segment, Revenue, Purchase Frequency, Last Purchase, Recommended Product
  - segment_summary.csv          -- pre-aggregated segment KPIs
  - monthly_revenue_trend.csv    -- for a revenue trend chart
  - dim_products.csv             -- product dimension table (for slicers)
"""

import pandas as pd

PROC = "/Users/macbook/Downloads/customer360/data/processed"
RAW = "/Users/macbook/Downloads/customer360/data/raw"
OUT = "/Users/macbook/Downloads/customer360/output"

df = pd.read_csv(f"{PROC}/customer_360.csv", parse_dates=["signup_date","last_purchase_date","first_purchase_date"])
sales = pd.read_csv(f"{RAW}/sales_orders.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------------------
# Main dashboard table -- rename to business-friendly labels matching the
# exact metrics requested: Customer Value, Churn Probability, Segment,
# Revenue, Purchase Frequency, Last Purchase, Recommended Product
# ---------------------------------------------------------------------------
dashboard = df.drop(columns=["segment"]).rename(columns={
    "clv_predicted": "customer_value",
    "monetary": "revenue",
    "frequency": "purchase_frequency",
    "last_purchase_date": "last_purchase",
    "segment_name": "segment",
})

dashboard_cols = [
    "customer_id","customer_name","region","country","customer_type",
    "segment","customer_value","clv_historical","revenue","purchase_frequency",
    "avg_order_value","recency_days","last_purchase","signup_date",
    "churn_probability","predicted_churn",
    "favorite_category","recommended_product","recommendation_score",
    "top_purchase_channel","total_sessions","avg_csat","total_tickets",
    "RFM_score","RFM_total","email_opt_in",
]
dashboard_cols = [c for c in dashboard_cols if c in dashboard.columns]
dashboard_final = dashboard[dashboard_cols].copy()

# Risk banding for easy Power BI conditional formatting / slicers
def risk_band(p):
    if p >= 0.7: return "High Risk"
    elif p >= 0.4: return "Medium Risk"
    else: return "Low Risk"
dashboard_final["churn_risk_band"] = dashboard_final["churn_probability"].apply(risk_band)

# Value tier for quick filtering
dashboard_final["value_tier"] = pd.qcut(
    dashboard_final["customer_value"].rank(method="first"), 4,
    labels=["Bronze","Silver","Gold","Platinum"]
)

dashboard_final.to_csv(f"{OUT}/customer_360_dashboard.csv", index=False)
print(f"customer_360_dashboard.csv: {dashboard_final.shape[0]:,} rows x {dashboard_final.shape[1]} cols")

# ---------------------------------------------------------------------------
# Segment summary (pre-aggregated KPIs, useful for a Power BI summary card page)
# ---------------------------------------------------------------------------
segment_summary = dashboard_final.groupby("segment").agg(
    customer_count=("customer_id","count"),
    avg_customer_value=("customer_value","mean"),
    total_revenue=("revenue","sum"),
    avg_churn_probability=("churn_probability","mean"),
    avg_purchase_frequency=("purchase_frequency","mean"),
    high_risk_customers=("churn_risk_band", lambda x: (x=="High Risk").sum()),
).round(2).reset_index().sort_values("total_revenue", ascending=False)
segment_summary.to_csv(f"{OUT}/segment_summary.csv", index=False)
print(f"\nsegment_summary.csv:\n{segment_summary}")

# ---------------------------------------------------------------------------
# Monthly revenue trend (for a time-series chart in the dashboard)
# ---------------------------------------------------------------------------
sales["order_month"] = sales["order_date"].dt.to_period("M").astype(str)
monthly_trend = sales.groupby("order_month").agg(
    revenue=("revenue","sum"),
    orders=("order_id","nunique"),
    unique_customers=("customer_id","nunique"),
).reset_index().sort_values("order_month")
monthly_trend.to_csv(f"{OUT}/monthly_revenue_trend.csv", index=False)
print(f"\nmonthly_revenue_trend.csv: {monthly_trend.shape[0]} months")

# ---------------------------------------------------------------------------
# Product dimension table (for slicers / product-level analysis)
# ---------------------------------------------------------------------------
dim_products = sales.groupby(["product_id","product_name","category"]).agg(
    total_revenue=("revenue","sum"),
    units_sold=("quantity","sum"),
    avg_price=("unit_price","mean"),
    times_recommended=("product_name", lambda x: 0),  # placeholder, filled below
).reset_index().drop(columns="times_recommended")

reco_counts = dashboard_final["recommended_product"].value_counts().rename_axis("product_name").reset_index(name="times_recommended")
dim_products = dim_products.merge(reco_counts, on="product_name", how="left")
dim_products["times_recommended"] = dim_products["times_recommended"].fillna(0).astype(int)
dim_products.to_csv(f"{OUT}/dim_products.csv", index=False)
print(f"\ndim_products.csv: {dim_products.shape[0]} products")

print("\nAll Power BI export files ready in /output")
