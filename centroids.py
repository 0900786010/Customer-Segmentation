import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
df = pd.read_csv("data/raw/customers_with_clusters.csv")
kmeans = joblib.load("models/kmeans_model.joblib")

centers = kmeans.cluster_centers_

# Extract original scale features
income = df["Annual Income (k$)"]
spending = df["Spending Score (1-100)"]
clusters = df["Cluster"]

plt.figure(figsize=(8,6))

# Plot customers
plt.scatter(income, spending, c=clusters, s=60, alpha=0.6)

# Plot cluster centroids
plt.scatter(centers[:,1], centers[:,2], c='red', s=300, marker='X')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Cluster Centers (Centroids)")
plt.grid(True)
plt.show()
