import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib

# ================================
# ✅ Load Data & Model
# ================================
df = pd.read_csv("data/raw/customers_with_clusters.csv")
kmeans = joblib.load("models/kmeans_model.joblib")

# ================================
# ✅ 1) Basic 2D Scatter Plot
# ================================
plt.figure(figsize=(8,6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=80
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (2D Scatter)")
plt.grid(True)
plt.show()

# ================================
# ✅ 2) Pair Plot
# ================================
sns.pairplot(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]],
    hue="Cluster",
    palette="tab10",
    diag_kind="kde"
)
plt.suptitle("Pair Plot of Customer Segments", y=1.02)
plt.show()

# ================================
# ✅ 3) 3D Scatter Plot
# ================================
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

x = df["Annual Income (k$)"]
y = df["Spending Score (1-100)"]
z = df["Age"]
clusters = df["Cluster"]

ax.scatter(x, y, z, c=clusters, s=100)
ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_zlabel("Age")
plt.title("3D Customer Segments")
plt.show()

# ================================
# ✅ 4) Cluster Centers Plot
# ================================
centers = kmeans.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(
    df["Annual Income (k$)"], df["Spending Score (1-100)"],
    c=df["Cluster"], s=70, alpha=0.6
)
# Centroids (centers)
plt.scatter(centers[:,1], centers[:,2], c='red', s=300, marker='X', label="Centroids")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Cluster Centers (Centroids)")
plt.legend()
plt.grid(True)
plt.show()

# ================================
# ✅ 5) Heatmap of Cluster Means
# ================================
cluster_means = df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()

plt.figure(figsize=(8,4))
sns.heatmap(cluster_means, annot=True, cmap="coolwarm")
plt.title("Cluster Feature Averages (Heatmap)")
plt.show()
