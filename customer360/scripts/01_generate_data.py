"""
01_generate_data.py
--------------------
Generates realistic SYNTHETIC data for a Retail Customer 360 project.
Tables are structured to mimic real SAP ECC / SAP BODS extract outputs,
so the ETL and documentation can honestly describe "SAP-style customer
master and sales data" in an interview.

Sources generated:
  - customers.csv          (mimics SAP KNA1 customer master)
  - sales_orders.csv       (mimics SAP VBAK/VBAP sales order header+line)
  - website_activity.csv   (e-commerce clickstream/session data)
  - support_tickets.csv    (customer service / CRM ticket data)
  - marketing.csv          (campaign exposure + response data)

Design choice: churn, CLV and segment are NOT random. They are generated
from an underlying "true" customer behavior so that later ML models
(logistic regression / random forest / xgboost / k-means) have genuine
signal to learn from -- this matters for a portfolio project because a
model trained on pure noise won't show meaningful metrics or feature
importances in an interview.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 5000
TODAY = datetime(2026, 8, 12)
SIGNUP_START = datetime(2021, 1, 1)

# ---------------------------------------------------------------------------
# Reference data (kept simple/curated instead of using an external lib,
# since this sandbox has no internet access to install Faker)
# ---------------------------------------------------------------------------
FIRST_NAMES = ["James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda",
    "David","Elizabeth","William","Barbara","Richard","Susan","Joseph","Jessica",
    "Thomas","Sarah","Charles","Karen","Ananya","Wei","Fatima","Carlos","Yuki",
    "Omar","Priya","Chen","Amara","Diego","Elena","Hiroshi","Layla","Noah",
    "Sofia","Liam","Mia","Ethan","Ava","Lucas"]
LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson","White",
    "Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker","Young","King"]

REGIONS = ["North America","Europe","APAC","Middle East","Latin America"]
COUNTRIES_BY_REGION = {
    "North America": ["USA","Canada","Mexico"],
    "Europe": ["Germany","UK","France","Spain","Italy","Netherlands"],
    "APAC": ["India","Japan","Singapore","Australia","South Korea"],
    "Middle East": ["Kuwait","UAE","Saudi Arabia","Qatar"],
    "Latin America": ["Brazil","Argentina","Chile"],
}
CUSTOMER_TYPES = ["Individual","Small Business","Enterprise"]
SALES_CHANNELS = ["Online","In-Store","Marketplace","B2B Direct"]

PRODUCT_CATALOG = [
    ("P001","Wireless Earbuds Pro","Electronics",89.99),
    ("P002","4K Smart TV 55in","Electronics",549.99),
    ("P003","Running Shoes","Apparel",79.99),
    ("P004","Winter Jacket","Apparel",129.99),
    ("P005","Espresso Machine","Home & Kitchen",249.99),
    ("P006","Air Fryer","Home & Kitchen",99.99),
    ("P007","Yoga Mat","Sports & Outdoors",29.99),
    ("P008","Mountain Bike","Sports & Outdoors",459.99),
    ("P009","Skincare Set","Beauty",59.99),
    ("P010","Perfume","Beauty",74.99),
    ("P011","Bluetooth Speaker","Electronics",49.99),
    ("P012","Laptop Backpack","Accessories",39.99),
    ("P013","Office Chair","Furniture",189.99),
    ("P014","Standing Desk","Furniture",349.99),
    ("P015","Board Game Set","Toys & Games",34.99),
    ("P016","Kids Building Blocks","Toys & Games",44.99),
    ("P017","Protein Powder","Health & Wellness",39.99),
    ("P018","Smart Watch","Electronics",199.99),
    ("P019","Denim Jeans","Apparel",64.99),
    ("P020","Cookware Set","Home & Kitchen",149.99),
]
prod_df = pd.DataFrame(PRODUCT_CATALOG, columns=["product_id","product_name","category","unit_price"])

SUPPORT_TICKET_TYPES = ["Delivery Delay","Product Defect","Billing Issue","Return Request",
    "Account Access","General Inquiry","Refund Request","Technical Support"]
MARKETING_CHANNELS = ["Email","SMS","Push Notification","Social Media Ad","Search Ad"]
CAMPAIGNS = ["Summer Sale 2025","Diwali Offer 2025","Black Friday 2025","New Year 2026",
    "Spring Refresh 2026","Loyalty Rewards Q2 2026","Flash Sale July 2026"]

def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=int(np.random.randint(0, max(delta, 1))))

# ---------------------------------------------------------------------------
# 1. CUSTOMERS (SAP KNA1-style customer master)
# ---------------------------------------------------------------------------
customer_ids = [f"C{100000+i}" for i in range(N_CUSTOMERS)]
regions = np.random.choice(REGIONS, N_CUSTOMERS, p=[0.35,0.25,0.20,0.10,0.10])
countries = [np.random.choice(COUNTRIES_BY_REGION[r]) for r in regions]
signup_dates = [random_date(SIGNUP_START, TODAY - timedelta(days=1)) for _ in range(N_CUSTOMERS)]
cust_types = np.random.choice(CUSTOMER_TYPES, N_CUSTOMERS, p=[0.75,0.18,0.07])

# Latent "true propensity" drivers -- not exposed directly, used to
# generate correlated downstream behavior (this is what gives the ML
# models real signal to find)
engagement_score = np.clip(np.random.beta(2, 2, N_CUSTOMERS), 0, 1)          # website/marketing engagement
value_score = np.clip(np.random.gamma(2, 1.4, N_CUSTOMERS) / 10, 0, 1)        # spending propensity
satisfaction_score_latent = np.clip(np.random.beta(3, 2, N_CUSTOMERS), 0, 1)  # support experience quality

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "first_name": np.random.choice(FIRST_NAMES, N_CUSTOMERS),
    "last_name": np.random.choice(LAST_NAMES, N_CUSTOMERS),
    "region": regions,
    "country": countries,
    "customer_type": cust_types,
    "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
    "preferred_channel": np.random.choice(SALES_CHANNELS, N_CUSTOMERS, p=[0.5,0.25,0.15,0.10]),
    "email_opt_in": np.random.choice([1,0], N_CUSTOMERS, p=[0.72,0.28]),
    "_engagement_latent": engagement_score,
    "_value_latent": value_score,
    "_satisfaction_latent": satisfaction_score_latent,
})

# ---------------------------------------------------------------------------
# 2. SALES ORDERS (SAP VBAK/VBAP-style order header + line item, flattened)
# ---------------------------------------------------------------------------
order_rows = []
order_counter = 500000

for idx, row in customers.iterrows():
    signup = datetime.strptime(row["signup_date"], "%Y-%m-%d")
    tenure_days = (TODAY - signup).days

    # number of orders driven by value_latent + engagement_latent + tenure
    base_rate = (row["_value_latent"] * 0.7 + row["_engagement_latent"] * 0.3)
    expected_orders = max(0.3, base_rate * (tenure_days / 90))  # orders per ~quarter scaled
    n_orders = np.random.poisson(lam=min(expected_orders, 40))

    # Recency behavior: lower value/engagement customers stop ordering earlier (churn signal)
    churn_propensity = 1 - (row["_value_latent"] * 0.5 + row["_engagement_latent"] * 0.5)
    last_active_frac = np.clip(1 - churn_propensity * np.random.uniform(0.3, 1.0), 0.02, 1.0)
    last_active_day_offset = int(tenure_days * (1 - last_active_frac))

    for _ in range(n_orders):
        order_counter += 1
        days_ago = np.random.randint(last_active_day_offset, tenure_days + 1) if tenure_days > 0 else 0
        order_date = TODAY - timedelta(days=int(np.clip(tenure_days - days_ago, 0, tenure_days)))
        n_items = np.random.choice([1,1,2,2,3,4], p=[0.35,0.25,0.2,0.1,0.06,0.04])
        chosen = prod_df.sample(n_items, replace=True)
        for _, p in chosen.iterrows():
            qty = np.random.choice([1,1,1,2,3], p=[0.55,0.2,0.15,0.06,0.04])
            discount_pct = np.random.choice([0,0,0,0.1,0.15,0.2], p=[0.55,0.15,0.1,0.1,0.06,0.04])
            unit_price = p["unit_price"]
            revenue = round(unit_price * qty * (1 - discount_pct), 2)
            order_rows.append({
                "order_id": f"SO{order_counter}",
                "customer_id": row["customer_id"],
                "order_date": order_date.strftime("%Y-%m-%d"),
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "quantity": qty,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "revenue": revenue,
                "sales_channel": np.random.choice(SALES_CHANNELS, p=[0.5,0.25,0.15,0.10]),
            })

sales_orders = pd.DataFrame(order_rows)

# ---------------------------------------------------------------------------
# 3. WEBSITE ACTIVITY
# ---------------------------------------------------------------------------
web_rows = []
session_counter = 900000
for idx, row in customers.iterrows():
    signup = datetime.strptime(row["signup_date"], "%Y-%m-%d")
    tenure_days = max((TODAY - signup).days, 1)
    n_sessions = np.random.poisson(lam=max(1, row["_engagement_latent"] * 25))
    for _ in range(n_sessions):
        session_counter += 1
        days_ago = np.random.randint(0, tenure_days)
        session_date = TODAY - timedelta(days=days_ago)
        pages_viewed = np.random.poisson(lam=3 + row["_engagement_latent"] * 6) + 1
        cart_add = np.random.choice([1,0], p=[min(0.05 + row["_value_latent"]*0.3, 0.6), None] and
                                     [min(0.05 + row["_value_latent"]*0.3, 0.6), 1 - min(0.05 + row["_value_latent"]*0.3, 0.6)])
        device = np.random.choice(["Mobile","Desktop","Tablet"], p=[0.6,0.32,0.08])
        web_rows.append({
            "session_id": f"WEB{session_counter}",
            "customer_id": row["customer_id"],
            "session_date": session_date.strftime("%Y-%m-%d"),
            "pages_viewed": pages_viewed,
            "time_on_site_min": round(np.random.gamma(2, 2) + pages_viewed * 0.5, 1),
            "device": device,
            "added_to_cart": int(cart_add),
            "traffic_source": np.random.choice(["Direct","Organic Search","Paid Search","Social","Email"], p=[0.25,0.3,0.2,0.15,0.1]),
        })

website_activity = pd.DataFrame(web_rows)

# ---------------------------------------------------------------------------
# 4. SUPPORT TICKETS
# ---------------------------------------------------------------------------
support_rows = []
ticket_counter = 700000
for idx, row in customers.iterrows():
    signup = datetime.strptime(row["signup_date"], "%Y-%m-%d")
    tenure_days = max((TODAY - signup).days, 1)
    # Lower satisfaction latent -> more tickets, worse resolution
    n_tickets = np.random.poisson(lam=max(0.1, (1 - row["_satisfaction_latent"]) * 3))
    for _ in range(n_tickets):
        ticket_counter += 1
        days_ago = np.random.randint(0, tenure_days)
        ticket_date = TODAY - timedelta(days=days_ago)
        resolution_hours = round(np.random.gamma(2, 8) * (1.5 - row["_satisfaction_latent"]), 1)
        csat = int(np.clip(np.random.normal(row["_satisfaction_latent"] * 5, 1), 1, 5))
        support_rows.append({
            "ticket_id": f"TKT{ticket_counter}",
            "customer_id": row["customer_id"],
            "ticket_date": ticket_date.strftime("%Y-%m-%d"),
            "ticket_type": np.random.choice(SUPPORT_TICKET_TYPES),
            "resolution_time_hours": resolution_hours,
            "csat_score": csat,
            "escalated": np.random.choice([1,0], p=[max(0.05,(1-row["_satisfaction_latent"])*0.3), 1-max(0.05,(1-row["_satisfaction_latent"])*0.3)]),
        })

support_tickets = pd.DataFrame(support_rows)

# ---------------------------------------------------------------------------
# 5. MARKETING
# ---------------------------------------------------------------------------
marketing_rows = []
mkt_counter = 800000
for idx, row in customers.iterrows():
    n_touches = np.random.poisson(lam=max(0.5, row["_engagement_latent"] * 8))
    for _ in range(n_touches):
        mkt_counter += 1
        campaign = np.random.choice(CAMPAIGNS)
        channel = np.random.choice(MARKETING_CHANNELS)
        converted = np.random.choice([1,0], p=[min(0.05 + row["_engagement_latent"]*row["_value_latent"]*0.5, 0.5),
                                                1 - min(0.05 + row["_engagement_latent"]*row["_value_latent"]*0.5, 0.5)])
        marketing_rows.append({
            "touch_id": f"MKT{mkt_counter}",
            "customer_id": row["customer_id"],
            "campaign_name": campaign,
            "channel": channel,
            "sent_date": random_date(SIGNUP_START, TODAY).strftime("%Y-%m-%d"),
            "opened": np.random.choice([1,0], p=[min(0.3 + row["_engagement_latent"]*0.5,0.9), 1-min(0.3+row["_engagement_latent"]*0.5,0.9)]),
            "converted": converted,
        })

marketing = pd.DataFrame(marketing_rows)

# ---------------------------------------------------------------------------
# Drop latent helper columns before saving (keep raw sources "clean" like
# real source systems -- latents were only used to generate correlated data)
# ---------------------------------------------------------------------------
customers_out = customers.drop(columns=["_engagement_latent","_value_latent","_satisfaction_latent"])

RAW = "/Users/macbook/Downloads/customer360/data/raw"
customers_out.to_csv(f"{RAW}/customers.csv", index=False)
sales_orders.to_csv(f"{RAW}/sales_orders.csv", index=False)
website_activity.to_csv(f"{RAW}/website_activity.csv", index=False)
support_tickets.to_csv(f"{RAW}/support_tickets.csv", index=False)
marketing.to_csv(f"{RAW}/marketing.csv", index=False)

print("Data generation complete.")
print(f"customers:         {len(customers_out):,} rows")
print(f"sales_orders:       {len(sales_orders):,} rows")
print(f"website_activity:   {len(website_activity):,} rows")
print(f"support_tickets:    {len(support_tickets):,} rows")
print(f"marketing:          {len(marketing):,} rows")
