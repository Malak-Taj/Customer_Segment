import pandas as pd
import numpy as np

def preprocess_customer_data(df):

    # This process includes:
    """
    - Aggregation
    - Category ratios
    - Category entropy
    - Log-transform of skewed numeric features
    """
    
    # Numeric aggregation per customer
    df_customer = df.groupby('User_Id').agg({
        'Customer_Age': 'min',
        'Trx_Vlu': 'sum',
        'Trx_Age': 'min',
        'Points': 'sum'
    }).reset_index()
    
    # Category counts per customer
    category_features = df.groupby(['User_Id', 'Category']).size().unstack(fill_value=0)
    
    # Join with original table
    df_customer = df_customer.join(category_features)
    
    cat_cols = category_features.columns
    
    # Normalize category counts → ratios
    df_customer[cat_cols] = df_customer[cat_cols].div(
        df_customer[cat_cols].sum(axis=1), axis=0
    ).fillna(0)

    # Category entropy
    df_customer['Category_Entropy'] = -(
        df_customer[cat_cols]
        .replace(0, np.nan)
        .apply(lambda x: x * np.log(x), axis=1)
        .sum(axis=1)
    ).fillna(0).abs()
    
    # Log-transform skewed numeric columns
    df_customer['Trx_Vlu'] = np.log1p(df_customer['Trx_Vlu'])
    df_customer['Points'] = np.log1p(df_customer['Points'])

    # save processed customer dataframe
    df_customer.to_parquet('Clean_data/Customer_Level_cleaned_Data.parquet', engine='pyarrow')
    
    return df_customer
