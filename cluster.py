# src/cluster.py
from sklearn.cluster import KMeans
import joblib
import os
from preprocess import preprocess

def run_clustering(k=4):
    X_scaled, df = preprocess()

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = labels

    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/kmeans_model.joblib")
    df.to_csv("data/raw/customers_with_clusters.csv", index=False)

    print(f"✅ Clustering done with k={k}. Output: data/raw/customers_with_clusters.csv")

if __name__ == "__main__":
    run_clustering()
