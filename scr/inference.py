import pandas as pd
import joblib
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def run_dbscan_inference(
    df_new: pd.DataFrame,
    scaler_path: str = "Models/scaler_pipeline.pkl",
    numeric_meta_path: str = "Metadata/numeric_columns.pkl",
    cat_meta_path: str = "Metadata/cat_cols.pkl",
    eps: float = 1.0,
    min_samples: int = 20,
    save_path: str = "Data/Clean_data/new_customers_with_clusters.parquet",
    top_n_merchants: int = 3
):
    # Load column metadata
    numeric_columns = joblib.load(numeric_meta_path)
    cat_cols = joblib.load(cat_meta_path)
    all_cols = numeric_columns + cat_cols

    # Ensure columns exist and correct order
    df_new = df_new[all_cols]

    # Load and apply scaling
    scaler_pipeline = joblib.load(scaler_path)
    X_scaled = scaler_pipeline.transform(df_new[all_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_cols)


    # Run DBSCAN clustering
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1
    )
    labels = dbscan.fit_predict(X_scaled_df.values)
    df_new["Cluster"] = labels


    # Profile clusters(top merchants)
    cluster_profile = df_new.groupby("Cluster")[cat_cols].mean()
    top_merchants = {}
    for cluster in cluster_profile.index:
        sorted_merchants = cluster_profile.loc[cluster].sort_values(ascending=False).index.tolist()
        top_merchants[cluster] = sorted_merchants[:top_n_merchants]

   
    # Assign nearest cluster & recommend merchants
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_scaled_df.values)

    recommended_list = []
    for i in range(X_scaled_df.shape[0]):
        dist, idx = nn.kneighbors([X_scaled_df.iloc[i].values])
        assigned_cluster = df_new.iloc[idx[0][0]]["Cluster"]
        recommended_list.append(top_merchants.get(assigned_cluster, []))

    df_new["Recommended_Merchants"] = recommended_list

    # Results
    print("Inference completed successfully!")
    print(f"Clusters found: {set(labels)}")
    print(f"Number of noise points (-1): {(labels == -1).sum()}")

    return df_new
