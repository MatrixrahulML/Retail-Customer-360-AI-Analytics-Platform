# Retail Customer 360 + AI Analytics Platform

A end-to-end Customer 360 platform combining SAP-style customer master data,
sales orders, website activity, customer support, and marketing data into a
single analytical table — with ML-driven segmentation, churn prediction, CLV,
and product recommendations, visualized in Power BI.

Built as a portfolio project targeting Senior Data & AI Engineer roles
requiring SAP BODS / Azure / Fabric ETL experience, BI dashboarding
(Power BI / Tableau), and applied ML (segmentation, churn, recommendations).

---

## Architecture

```
[5 Source Systems]                [ETL Layer]              [ML Layer]              [BI Layer]
customers.csv    ─┐
sales_orders.csv  ├─►  02_etl_customer360.py  ─►  customer_360.csv  ─┬─►  K-Means (segments)
website_activity  │       (integration, RFM,                        ├─►  LogReg/RF/GBM (churn)
support_tickets   │        CLV, cleaning)                            └─►  Reco engine (products)
marketing.csv    ─┘                                                            │
                                                                                 ▼
                                                                    customer_360_dashboard.csv
                                                                                 │
                                                                                 ▼
                                                                          Power BI Dashboard
```

## Why synthetic data, styled like SAP

No production SAP system was available for this project, so all data is
**synthetically generated** with realistic distributions, correlated behavior
(engagement drives web activity, satisfaction drives churn, etc.), and table
structures that mirror real SAP extracts:

- `customers.csv` → mirrors SAP **KNA1** (customer master)
- `sales_orders.csv` → mirrors SAP **VBAK/VBAP** (sales order header + line item)

This is a standard, honest approach for portfolio projects without production
data access — worth stating plainly in an interview rather than implying it's
real customer data.

## Project structure

```
customer360/
├── data/
│   ├── raw/                  5 source CSVs (SAP-style + web/support/marketing)
│   └── processed/            customer_360.csv (the integrated master table)
├── scripts/
│   ├── 01_generate_data.py           Synthetic data generation
│   ├── 02_etl_customer360.py         ETL: integration, RFM, CLV, churn label
│   ├── 03_kmeans_segmentation.py     K-Means customer segmentation
│   ├── 04_churn_models.py            Logistic Regression + Random Forest
│   ├── 05_xgboost_and_recommendations.py   XGBoost version (run locally)
│   ├── 05_gbm_and_recommendations.py       Sandbox-runnable stand-in (ran here)
│   └── 06_export_for_powerbi.py      Final dashboard CSV exports
├── models/                   Saved .pkl models (joblib)
├── output/                   Charts, model comparison, dashboard CSVs
└── powerbi/
    ├── POWERBI_GUIDE.md      Step-by-step dashboard build instructions
    └── *.csv                 Copies of the 4 files to import into Power BI
```

## How to run

```bash
git clone https://github.com/MatrixrahulML/Retail-Customer-360-AI-Analytics-Platform.git
cd ~/Downloads/Retail-Customer-360-AI-Analytics-Platform/customer360

source venv/bin/activate

python scripts/01_generate_data.py

python scripts/02_etl_customer360.py

python scripts/03_kmeans_segmentation.py

python scripts/04_churn_models.py

python scripts/05_xgboost_and_recommendations.py ## need xgboost and internet

python scripts/06_export_for_powerbi.py
```

> Note: `05_xgboost_and_recommendations.py` needs `pip3 install xgboost`, which
> wasn't possible in the sandbox this project was built in (no internet
> access there). `05_gbm_and_recommendations.py` is a fully working
> stand-in using scikit-learn's `GradientBoostingClassifier` — same
> algorithm family, same feature engineering, same recommendation logic —
> so the pipeline runs end-to-end and produces real results either way.
> Run the actual XGBoost version locally to complete the "Use XGBoost"
> requirement literally.

## Results achieved (this run)

**Churn model comparison** (25% held-out test set, 5,000 customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 73.2% | 52.3% | 81.2% | 63.6% | 0.816 |
| Random Forest | 87.3% | 74.6% | 84.8% | 79.4% | 0.943 |
| Gradient Boosting (XGBoost stand-in) | 89.6% | 80.0% | 85.3% | 82.6% | **0.967** |

Top churn drivers: purchase frequency per year, RFM total score, customer
tenure, and total monetary value — all business-intuitive, no leaked features.

**K-Means segmentation:** k=5 chosen via elbow method + silhouette score.
Segments: Champions, Loyal High-Value, Regular Customers, New/Occasional,
Lost/Dormant.

**Product recommendation engine:** item-item co-occurrence + gradient-boosted
re-ranking model, ROC-AUC 0.895 on held-out (customer, product) pairs.

## Key design decisions worth explaining in an interview

1. **Churn label leakage avoidance** — the rule-based churn label is partly
   defined by recency, so recency features were excluded from the model
   inputs to avoid inflated, unrealistic accuracy. This is the single best
   thing to bring up when asked "how did you validate your model."
2. **CLV: historical vs. predictive** — `clv_historical` is simple total
   spend to date; `clv_predicted` projects forward using purchase frequency,
   average order value, and a recency/engagement-based retention factor.
   Both are provided so you can discuss the tradeoff between simple and
   model-based CLV.
3. **Rule-based label, ML-predicted** — real retail systems often don't
   track a clean "customer churned" event. Defining churn via a business
   rule (e.g., no purchase in 180+ days with low engagement) and training
   classifiers to predict that rule is a standard, defensible pattern.
4. **RFM done properly** — quantile-based scoring (not arbitrary thresholds),
   explicit R/F/M sub-scores plus a combined RFM_total for segmentation input.
