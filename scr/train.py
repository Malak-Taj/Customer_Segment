import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN

def main():
    # Load customer data
    df_customer = pd.read_parquet(
        "Data/Clean_data/Customer_Level_cleaned_Data.parquet",
        engine='pyarrow'
    )


    # Load scaling pipeline
    scaling_pipeline = joblib.load("Models/scaler_pipeline.pkl")


    # Create DBSCAN model
    dbscan = DBSCAN(
        eps=1,
        min_samples=20,
        metric="euclidean",
        n_jobs=-1
    )

    # Build clustering pipeline
    clustering_pipeline = Pipeline([
        ("scaling", scaling_pipeline),
        ("clustering", dbscan)
    ])

 
    # Fit and predict clusters
    labels = clustering_pipeline.fit_predict(df_customer)
    df_customer["Cluster"] = labels

 
    # Save clustered data
    df_customer.to_parquet("Data/Clean_data/customer_data_with_clusters.parquet",  index=False)


    # Save clustering pipeline
    joblib.dump(clustering_pipeline, "Models/dbscan_pipeline.pkl")


    # Summary
    print("DBSCAN pipeline trained & saved correctly")
    print(f"Clusters found: {set(labels)}")
    print(f"Number of noise points (-1): {(labels == -1).sum()}")

if __name__ == "__main__":
    main()
