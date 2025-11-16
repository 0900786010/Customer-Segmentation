import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with clusters
df = pd.read_csv("data/raw/customers_with_clusters.csv")

# Select columns for the pairplot
cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]

sns.pairplot(df[cols], hue="Cluster", palette="tab10", diag_kind="kde")

plt.suptitle("Pair Plot of Customer Segments", y=1.02)
plt.show()
