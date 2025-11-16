import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("data/raw/customers_with_clusters.csv")

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

x = df["Annual Income (k$)"]
y = df["Spending Score (1-100)"]
z = df["Age"]
clusters = df["Cluster"]

scatter = ax.scatter(x, y, z, c=clusters, s=80)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_zlabel("Age")

plt.title("3D Customer Segments")
plt.show()
