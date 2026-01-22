# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from scr.inference import run_dbscan_inference  # inference function

# paths
scaler_pipeline_path = "Models/scaler_pipeline.pkl"
numeric_meta_path = "Metadata/numeric_columns.pkl"
cat_meta_path = "Metadata/cat_cols.pkl"
clustered_data_path = "Data/Clean_data/customer_data_with_clusters.parquet"

# Load Data 
df_clustered = pd.read_parquet(clustered_data_path)
numeric_columns = joblib.load(numeric_meta_path)
cat_cols = joblib.load(cat_meta_path)

# Prepare top merchants dictionary
cluster_profile = df_clustered.groupby('Cluster')[cat_cols].mean()
top_merchants = {}
for cluster in cluster_profile.index:
    sorted_merchants = cluster_profile.loc[cluster].sort_values(ascending=False).index.tolist()
    top_merchants[cluster] = sorted_merchants[:3]

# Streamlit UI
st.title("Customer Cluster Prediction & Merchant Recommendation")
st.write("Enter customer information:")

#User Inputs 
customer_age_months = st.number_input("Customer Age (days)", value=30)
trx_age = st.number_input("Transaction Age (days)", value=12)
trx_vlu = st.number_input("Transaction Value ($)", value=5000)

if st.button("Predict Cluster & Recommend Merchants"):

    #  Find nearest existing customer for missing features
    nn_features = ['Customer_Age', 'Trx_Age', 'Trx_Vlu']
    X_train = df_clustered[nn_features].values

    new_customer_features = [[customer_age_months, trx_age, trx_vlu]]
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(X_train)
    dist, idx = nn_model.kneighbors(new_customer_features)
    nearest_customer = df_clustered.iloc[idx[0][0]]

    #  Build complete input
    df_new = pd.DataFrame([{
        'Customer_Age': customer_age_months,
        'Trx_Age': trx_age,
        'Trx_Vlu': trx_vlu,
        'Category_Entropy': nearest_customer['Category_Entropy'],
        'Electronics': nearest_customer['Electronics'],
        'F&B': nearest_customer['F&B'],
        'Fashion': nearest_customer['Fashion'],
        'Grocery': nearest_customer['Grocery'],
        'Health & Beauty': nearest_customer['Health & Beauty'],
        'Other': nearest_customer['Other']
    }])

    #  Run Inference 
    clustered_df = run_dbscan_inference(
        df_new=df_new,
        scaler_path=scaler_pipeline_path,
        numeric_meta_path=numeric_meta_path,
        cat_meta_path=cat_meta_path,
        eps=1.0,
        min_samples=20
    )

    assigned_cluster = clustered_df["Cluster"].iloc[0]
    st.success(f"Assigned Cluster: {int(assigned_cluster)}")

    # Top recommended merchants
    recommended = top_merchants.get(assigned_cluster, [])
    st.subheader("Top 3 Recommended Merchants:")
    for i, merchant in enumerate(recommended, 1):
        st.write(f"{i}. {merchant}")
