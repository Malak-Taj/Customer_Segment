# Customer Segmentation & Merchant Recommendation App

This is a **Streamlit web app** that predicts the **customer cluster** and recommends the **top 3 merchants** based on customer transaction data. It uses **DBSCAN clustering** on historical customer data and **nearest neighbor assignment** for new customers.

---

## Features

- Input **Customer Age**, **Transaction Age**, and **Transaction Value**.
- Automatically computes **Points** as `Transaction Value × 10`.
- Predicts the **customer cluster** using a pre-trained DBSCAN model.
- Recommends the **top 3 merchants** for the assigned cluster.
- Automatically fills missing merchant/category values from the nearest historical customer.

---

## Folder Structure

Customer_segmentation/
├── app/
│ └── app.py # Streamlit app
├── Data/
│ └── Clean_data/
│ ├── Customer_Level_cleaned_Data.parquet
│ └── customer_data_with_clusters.parquet
├── Models/
│ ├── scaler_pipeline.pkl
│ └── dbscan_pipeline.pkl
├── Metadata/
│ ├── numeric_columns.pkl
│ └── cat_cols.pkl
├── scr/
│ ├── inference.py # DBSCAN inference function
│ └── scaling.py # Scaling pipeline function
└── README.md
