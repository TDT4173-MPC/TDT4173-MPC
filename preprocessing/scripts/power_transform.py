import pandas as pd
import sys
from sklearn.preprocessing import PowerTransformer, StandardScaler

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Remove the rows where the target is nan
    try:
        df = df[df['pv_measurement'].notna()]
    except KeyError:
        pass

    # Instantiate the PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    ss = StandardScaler()

    # Identify numerical columns for transformation (excluding binary/categorical)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if 'pv_measurement' in numeric_cols:
        numeric_cols = numeric_cols.drop('pv_measurement')  # Exclude target variable if it exists

    # Standardize the data first
    df[numeric_cols] = ss.fit_transform(df[numeric_cols])

    # Apply the Yeo-Johnson transformation with fallback
    for col in numeric_cols:
        try:
            df[col] = pt.fit_transform(df[[col]])
        except Exception as e:
            print(f"Transformation failed for {col}: {e}")
            # Fallback: skip the transformation for this column or handle differently

    # Save the handled data back to the same path
    df.to_parquet(input_file, index=False)
    print(f"Yeo-Johnson transformation applied to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)