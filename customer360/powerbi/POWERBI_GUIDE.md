# Power BI Dashboard — Build Guide

## 1. Files to import
Open Power BI Desktop → Get Data → Text/CSV → import all four files from `/output`:

| File | Purpose |
|---|---|
| `customer_360_dashboard.csv` | Main fact table — one row per customer |
| `segment_summary.csv` | Pre-aggregated segment KPIs (for summary cards) |
| `monthly_revenue_trend.csv` | Revenue over time (line chart) |
| `dim_products.csv` | Product dimension (slicers, recommendation analysis) |

No relationships are strictly required (each table is self-contained), but if you want
cross-filtering by product, create a relationship:
`customer_360_dashboard[recommended_product]` → `dim_products[product_name]` (many-to-one).

## 2. Recommended page layout

### Page 1 — Executive Overview
- **Cards (top row):** Total Revenue, Total Customers, Avg Churn Probability, Avg Customer Value
  - Total Revenue = `SUM(customer_360_dashboard[revenue])`
  - Avg Churn Probability = `AVERAGE(customer_360_dashboard[churn_probability])`
- **Line chart:** Revenue trend — `monthly_revenue_trend[order_month]` (axis) vs `revenue` (values)
- **Donut chart:** Customers by Segment — `segment` (legend) vs count of `customer_id`
- **Bar chart:** Revenue by Segment — from `segment_summary.csv`
- **Table/Matrix:** `segment_summary.csv` as a KPI summary grid

### Page 2 — Customer 360 Explorer (the core requested view)
Build a table/matrix visual with these exact columns from `customer_360_dashboard.csv`:

| Requested Field | Column in file |
|---|---|
| Customer Value | `customer_value` |
| Churn Probability | `churn_probability` (format as %) |
| Segment | `segment` |
| Revenue | `revenue` |
| Purchase Frequency | `purchase_frequency` |
| Last Purchase | `last_purchase` |
| Recommended Product | `recommended_product` |

- Add slicers: `segment`, `churn_risk_band`, `value_tier`, `region`, `favorite_category`
- Conditional formatting on `churn_probability`: red (>0.7) → yellow (0.4–0.7) → green (<0.4)
- Sort default by `customer_value` descending to surface top customers first

### Page 3 — Churn Analysis
- **Bar chart:** Customer count by `churn_risk_band`
- **Scatter chart:** `customer_value` (x) vs `churn_probability` (y), colored by `segment`
  — this immediately shows "high value + high churn risk" customers, the most
  actionable retention targets
- Import `rf_feature_importance.png` / `gbm_feature_importance.png` as an image visual,
  or recreate as a bar chart if you export feature importances to CSV
- Import `roc_curves.png` and `confusion_matrices.png` as image visuals to show model
  validation (great for demonstrating rigor in an interview walkthrough)

### Page 4 — Segmentation (RFM)
- **Scatter/bubble chart:** `frequency` (x) vs `recency_days` (y), bubble size = `revenue`,
  color = `segment`
- **Table:** `segment_profiles.csv` for the underlying cluster math
- Import `elbow_chart.png` to show how k=5 was chosen

### Page 5 — Product & Recommendations
- **Bar chart:** `dim_products[times_recommended]` by `product_name` — shows which
  products the model recommends most
- **Table:** top revenue products from `dim_products.csv`
- **Matrix:** `recommended_product` vs `segment` (count) — shows which segments get
  recommended which products

## 3. Suggested DAX measures

```
Total Revenue = SUM(customer_360_dashboard[revenue])

Avg Churn Probability = AVERAGE(customer_360_dashboard[churn_probability])

High Risk Customers = 
    CALCULATE(
        COUNTROWS(customer_360_dashboard),
        customer_360_dashboard[churn_risk_band] = "High Risk"
    )

Revenue at Risk = 
    CALCULATE(
        SUM(customer_360_dashboard[revenue]),
        customer_360_dashboard[churn_risk_band] = "High Risk"
    )

Avg Customer Lifetime Value = AVERAGE(customer_360_dashboard[customer_value])

Churn Rate % = 
    DIVIDE(
        CALCULATE(COUNTROWS(customer_360_dashboard), customer_360_dashboard[predicted_churn] = 1),
        COUNTROWS(customer_360_dashboard)
    )
```

`Revenue at Risk` is a strong metric to highlight — it translates the churn model
directly into a dollar figure business stakeholders care about, which is exactly the
kind of framing that lands well in interviews.

## 4. Talking points for the interview
- Explain the ETL: 5 source systems (styled like SAP customer master + sales orders,
  plus web/support/marketing) integrated into one customer-grain table.
- Explain the churn label: rule-based ground truth (since real cancellation events
  aren't tracked in retail), then supervised models learn to predict it — a common,
  defensible real-world pattern.
- Explain the leakage decision: recency-based fields were deliberately excluded from
  model features because the label itself is partly recency-based — shows ML maturity.
- Walk through model comparison: Logistic Regression (baseline/interpretable) →
  Random Forest (feature importance) → Gradient Boosted Trees/XGBoost (best performer),
  a natural "start simple, add complexity" narrative.
- K-Means segmentation: explain elbow method + silhouette score for choosing k=5,
  and how clusters were named using value + recency composite ranking.
- Recommendations: item-affinity + ML re-ranking, explainable as "customers who buy
  similar things get similar suggestions, refined by purchase propensity."
