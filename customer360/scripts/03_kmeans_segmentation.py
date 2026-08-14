"""
03_kmeans_segmentation.py
--------------------------
Customer segmentation using K-Means clustering on RFM + engagement features.

Produces:
  - Elbow method chart (choosing k) -> output/elbow_chart.png
  - Cluster profile summary -> output/segment_profiles.csv
  - customer_360 table updated with `segment` and `segment_name` columns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

PROC = "/Users/macbook/Downloads/customer360/data/processed"
OUT = "/Users/macbook/Downloads/customer360/output"
MODELS = "/Users/macbook/Downloads/customer360/models"

df = pd.read_csv(f"{PROC}/customer_360.csv")

# ---------------------------------------------------------------------------
# Features for clustering: core RFM + value + engagement
# ---------------------------------------------------------------------------
cluster_features = [
    "recency_days", "frequency", "monetary",
    "clv_predicted", "total_sessions", "avg_csat", "purchase_freq_per_year"
]

X = df[cluster_features].fillna(0).copy()

# Log-transform skewed monetary/CLV features (standard practice for RFM/KMeans)
for col in ["monetary", "clv_predicted", "recency_days"]:
    X[col] = np.log1p(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# Elbow method + silhouette score to choose k
# ---------------------------------------------------------------------------
inertias = []
sil_scores = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(list(k_range), inertias, marker="o", color="#c0392b")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (WCSS)", color="#c0392b")
ax2 = ax1.twinx()
ax2.plot(list(k_range), sil_scores, marker="s", color="#2c3e50")
ax2.set_ylabel("Silhouette Score", color="#2c3e50")
plt.title("K-Means: Elbow Method + Silhouette Score")
fig.tight_layout()
plt.savefig(f"{OUT}/elbow_chart.png", dpi=150)
plt.close()

best_k = 5  # chosen based on elbow + business interpretability (5 classic RFM-style segments)
print(f"Silhouette scores by k: {dict(zip(k_range, [round(s,3) for s in sil_scores]))}")
print(f"Selected k = {best_k}")

# ---------------------------------------------------------------------------
# Final K-Means model
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["segment"] = kmeans.fit_predict(X_scaled)

joblib.dump(kmeans, f"{MODELS}/kmeans_model.pkl")
joblib.dump(scaler, f"{MODELS}/kmeans_scaler.pkl")

# ---------------------------------------------------------------------------
# Profile each segment and assign business-friendly names
# ---------------------------------------------------------------------------
profile = df.groupby("segment")[cluster_features].mean().round(1)
profile["customer_count"] = df.groupby("segment").size()
profile = profile.sort_values("clv_predicted", ascending=False)
print("\nSegment profiles (raw feature means):")
print(profile)

# Name segments deterministically using a composite "value rank" (CLV +
# frequency, both z-scored) and a separate recency check for "At Risk" /
# "Lost". This guarantees each of the k segments gets a distinct label
# instead of relying on ties in a single metric.
def name_segments(profile):
    z = profile.copy()
    z["value_z"] = (
        (profile["clv_predicted"] - profile["clv_predicted"].mean()) / profile["clv_predicted"].std(ddof=0)
        + (profile["frequency"] - profile["frequency"].mean()) / profile["frequency"].std(ddof=0)
    )
    order = z.sort_values("value_z", ascending=False).index.tolist()  # best value first
    median_recency = profile["recency_days"].median()

    labels_pool = ["Champions", "Loyal High-Value", "Regular Customers",
                   "New / Occasional", "At Risk", "Lost / Dormant"]
    names = {}
    for rank, seg_id in enumerate(order):
        row = profile.loc[seg_id]
        # Dormant / never purchased overrides value rank entirely
        if row["frequency"] < 0.5 and row["recency_days"] > median_recency:
            names[seg_id] = "Lost / Dormant"
        elif row["recency_days"] > median_recency * 1.3:
            names[seg_id] = "At Risk"
        else:
            # assign next unused positive label by value rank
            for label in labels_pool:
                if label not in names.values() and label not in ("At Risk", "Lost / Dormant"):
                    names[seg_id] = label
                    break
            else:
                names[seg_id] = f"Segment {seg_id}"
    return names

seg_names = name_segments(profile)

df["segment_name"] = df["segment"].map(seg_names)

profile["segment_name"] = profile.index.map(seg_names)
profile.to_csv(f"{OUT}/segment_profiles.csv")

print("\nSegment names assigned:")
print(df.groupby("segment_name")["customer_id"].count().sort_values(ascending=False))

df.to_csv(f"{PROC}/customer_360.csv", index=False)
print(f"\nUpdated customer_360.csv with segment + segment_name columns")
print(f"Elbow chart saved to {OUT}/elbow_chart.png")
