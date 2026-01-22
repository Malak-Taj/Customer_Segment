import pandas as pd
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def main():
    # Load clustered customer data
    df_customer = pd.read_parquet( "Data/Clean_data/customer_data_with_clusters.parquet", engine='pyarrow')

    # Load column metadata
    numeric_columns = joblib.load("Metadata/numeric_columns.pkl")
    cat_cols = joblib.load("Metadata/cat_cols.pkl")
    all_cols = numeric_columns + cat_cols

    # Extract features and labels
    X = df_customer[all_cols].values
    labels = df_customer["Cluster"].values

    # Ignore noise points for metrics
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        silhouette = silhouette_score(X[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        calinski_harabasz = calinski_harabasz_score(X[mask], labels[mask])

        print("Clustering evaluation metrics:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    else:
        print("Not enough clusters (excluding noise) to calculate metrics.")

    # Summary of clusters
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution (including noise -1):")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} points")

if __name__ == "__main__":
    main()
