# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess():
    # read csv
    df = pd.read_csv("data/raw/customers.csv")

    # select features used for clustering
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    # simple missing value handling
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")

    return X_scaled, df

if __name__ == "__main__":
    X_scaled, df = preprocess()
    print("✅ Preprocessing completed. Rows:", df.shape[0])
