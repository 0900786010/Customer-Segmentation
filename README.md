# 🧩 Customer Segmentation using KMeans Clustering

This project performs **Customer Segmentation** using **KMeans Clustering** on a customer dataset.  
The goal is to group customers based on their **Age**, **Annual Income**, and **Spending Score** to help businesses understand customer behavior and target the right audience.

This project was completed as part of a **Data Science / Machine Learning internship task**.

---

## ✅ Project Overview

Customer segmentation helps businesses answer:
- Who are my high-spending customers?
- Which group needs discount offers?
- Who is rich but spends less?
- Which customers respond well to promotions?

Using **unsupervised learning (KMeans)**, we automatically create customer groups (clusters).

---

## ✅ Features of This Project

✔ Preprocessing (scaling + cleaning)  
✔ KMeans clustering  
✔ Saving model + scaler  
✔ Multiple visualizations (all in one command):
- 2D Scatter Plot  
- Pair Plot  
- 3D Plot  
- Centroid Plot  
- Cluster Means Heatmap  

✔ Easy-to-understand insights for each customer segment  
✔ Clean folder structure  
✔ Beginner-friendly code  

---

## ✅ Folder Structure

customer-segmentation/
│
├── data/
│ └── raw/
│ ├── customers.csv
│ └── customers_with_clusters.csv
│
├── models/
│ ├── scaler.joblib
│ └── kmeans_model.joblib
│
├── src/
│ ├── preprocess.py
│ ├── cluster.py
│ ├── visualize.py
│ └── all_visuals.py
│
└── README.md

---

## ✅ How It Works (Simple Explanation)

1. Load customer data  
2. Scale features (Age, Income, Spending Score)  
3. Apply **KMeans** to create clusters  
4. Save model + dataset with cluster labels  
5. Visualize customer groups

---

## ✅ How to Run the Project

### ✅ 1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### ✅ 2. Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

### ✅ 3. Run Clustering
python src/cluster.py

### ✅ 4. Run All Visualizations at Once
python src/all_visuals.py

## ✅ Visual Outputs

Your visualizations include:

- **Customer segments (2D)**  
- **Pair Plot (age, income, spending)**  
- **3D customer clusters**  
- **Cluster centroids plot**  
- **Cluster heatmap**

## ✅ Cluster Insights (Interpretation)

| Cluster | Characteristics | Business Insight |
|--------|----------------|------------------|
| **0** | Low income, low spending | Needs discounts or low-cost offers |
| **1** | Low income, high spending | Highly active buyers—good for loyalty offers |
| **2** | High income, low spending | Rich but low engagement—target premium promotions |
| **3** | High income, high spending | Best customers—ideal for premium and long-term offers |

## ✅ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Matplotlib  
- Seaborn  
- Joblib  

## ✅ Author

**Arfat Abid**  
Data Science / Machine Learning  
