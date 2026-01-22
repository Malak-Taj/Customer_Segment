import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import os

def build_scaling_pipeline(df_customer):
    # Columns
    numeric_columns = ['Customer_Age', 'Trx_Age', 'Trx_Vlu', 'Category_Entropy']
    cat_cols = ['Electronics','F&B','Fashion','Grocery','Health & Beauty','Other']

    all_cols = numeric_columns + cat_cols

    # Make directories if not exist
    os.makedirs('Metadata', exist_ok=True)
    os.makedirs('Models', exist_ok=True)

    # Save column metadata
    joblib.dump(numeric_columns, 'Metadata/numeric_columns.pkl')
    joblib.dump(cat_cols, 'Metadata/cat_cols.pkl')

    # Numeric pipeline
    num_pipe = Pipeline([('scaler', RobustScaler())])

    # ColumnTransformer
    pre_scaling = ColumnTransformer(
        transformers=[('numeric', num_pipe, numeric_columns)],
        remainder='passthrough'
    )

    # Fit + transform
    X_scaled = pre_scaling.fit_transform(df_customer[all_cols])

    # SAVE scaling pipeline
    joblib.dump(pre_scaling, 'Models/scaler_pipeline.pkl')

    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_cols)
    return X_scaled_df


if __name__ == "__main__":
    # Correct path relative to project root
    df_customer = pd.read_parquet('Data/Clean_data/Customer_Level_cleaned_Data.parquet', engine='pyarrow')
    X_scaled_df = build_scaling_pipeline(df_customer)
    print("Scaling pipeline built and saved successfully!")
    print("Scaled data preview:")
    print(X_scaled_df.head())
