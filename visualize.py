# src/visualize.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/customers_with_clusters.csv")

plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
            c=df["Cluster"], s=60)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (by Income vs Spending)")
plt.grid(True)
plt.show()
