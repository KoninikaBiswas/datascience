"""
Week 3: Unsupervised Learning and Clustering Analysis
Dataset: UCI Online Retail
Method: RFM customer segmentation + K-Means, with optional hierarchical clustering.

Input:
- online_retail_cleaned.csv (preferred, from Week 1), OR
- Online Retail.xlsx

Outputs:
- week3_outputs/clustered_customers.csv
- week3_outputs/cluster_profile.csv
- week3_outputs/k_selection.csv
- week3_outputs/*.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CSV_FILE = Path("online_retail_cleaned.csv")
XLSX_FILE = Path("Online Retail.xlsx")
OUT = Path("week3_outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
if CSV_FILE.exists():
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.read_excel(XLSX_FILE)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# Cancellation flag works for raw UCI data.
if "IsCancellation" not in df.columns:
    df["IsCancellation"] = (
        df["InvoiceNo"].astype("string").str.upper().str.startswith("C", na=False)
    )

# ---------------------------------------------------------
# 2. PREPARE POSITIVE CUSTOMER TRANSACTIONS
# ---------------------------------------------------------
sales = df[
    (~df["IsCancellation"]) &
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0) &
    (df["InvoiceDate"].notna()) &
    (df["CustomerID"].notna())
].copy()

sales["SalesAmount"] = sales["Quantity"] * sales["UnitPrice"]

# Use one day after the maximum transaction date as the reference date.
reference_date = sales["InvoiceDate"].max() + pd.Timedelta(days=1)

# ---------------------------------------------------------
# 3. CREATE RFM FEATURES
# ---------------------------------------------------------
rfm = sales.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("SalesAmount", "sum")
).reset_index()

print("Customers available for clustering:", len(rfm))
print(rfm.describe())

# ---------------------------------------------------------
# 4. REMOVE NON-POSITIVE / INVALID VALUES
# ---------------------------------------------------------
rfm = rfm[
    (rfm["Recency"] >= 0) &
    (rfm["Frequency"] > 0) &
    (rfm["Monetary"] > 0)
].copy()

# ---------------------------------------------------------
# 5. REDUCE SKEWNESS
# ---------------------------------------------------------
features = ["Recency", "Frequency", "Monetary"]
X = np.log1p(rfm[features])

# ---------------------------------------------------------
# 6. STANDARDIZE FEATURES
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 7. SELECT NUMBER OF CLUSTERS
# ---------------------------------------------------------
rows = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    rows.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(X_scaled, labels)
    })

selection = pd.DataFrame(rows)
selection.to_csv(OUT / "k_selection.csv", index=False)
print("\nK selection:\n", selection)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(selection["k"], selection["inertia"], marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of clusters (k)")
ax[0].set_ylabel("Inertia")

ax[1].plot(selection["k"], selection["silhouette"], marker="o")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("Number of clusters (k)")
ax[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig(OUT / "01_k_selection.png", dpi=200)
plt.close()

# Choose the k with the highest silhouette score.
best_k = int(selection.loc[selection["silhouette"].idxmax(), "k"])
print("Selected k:", best_k)

# ---------------------------------------------------------
# 8. K-MEANS MODEL
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------
# 9. CLUSTER PROFILE
# ---------------------------------------------------------
profile = rfm.groupby("Cluster")[features].agg(
    ["count", "mean", "median", "min", "max"]
)
profile.to_csv(OUT / "cluster_profile.csv")

simple_profile = rfm.groupby("Cluster")[features].mean()
simple_profile["Customers"] = rfm["Cluster"].value_counts().sort_index()
simple_profile.to_csv(OUT / "cluster_profile_simple.csv")

print("\nCluster profile:\n", simple_profile)

# ---------------------------------------------------------
# 10. PCA VISUALIZATION
# ---------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Cluster": rfm["Cluster"].astype(str)
})

plt.figure(figsize=(9,6))
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster",
                palette="tab10", alpha=.7, s=45)
plt.title("Customer Clusters in 2D PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(OUT / "02_pca_clusters.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 11. CLUSTER SIZE
# ---------------------------------------------------------
size = rfm["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8,5))
sns.barplot(x=size.index.astype(str), y=size.values)
plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig(OUT / "03_cluster_sizes.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 12. RFM PROFILE HEATMAP
# ---------------------------------------------------------
mean_profile = rfm.groupby("Cluster")[features].mean()
z_profile = (mean_profile - mean_profile.mean()) / mean_profile.std()

plt.figure(figsize=(7,5))
sns.heatmap(z_profile, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Standardized Cluster RFM Profile")
plt.xlabel("RFM Feature")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(OUT / "04_rfm_profile_heatmap.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 13. OPTIONAL HIERARCHICAL CLUSTERING
# ---------------------------------------------------------
hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
rfm["HierarchicalCluster"] = hier.fit_predict(X_scaled)

hier_sil = silhouette_score(X_scaled, rfm["HierarchicalCluster"])
print("Hierarchical silhouette score:", hier_sil)

# Save final customer segmentation.
rfm.to_csv(OUT / "clustered_customers.csv", index=False)

print("\nCompleted. Outputs saved to:", OUT.resolve())
