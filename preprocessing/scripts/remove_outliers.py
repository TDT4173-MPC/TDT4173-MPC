import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

def remove_outliers(df):
    """
    Remove outliers from a DataFrame using linear regression.
    """
    target_col = 'pv_measurement'
    
    columns = ['direct_rad:W', 'sun_elevation:d', 'diffuse_rad:W']

    for column in columns:

        if df.shape[0] == 0:  # Check if DataFrame is empty
            print(f"Warning: No data left when processing column '{column}'. Exiting early.")
            return df
        
        # Reshape data for linear regression
        X = df[column].values.reshape(-1, 1)
        y = df[target_col].values

        # Fit linear regression
        model = LinearRegression().fit(X, y)

        # Compute residuals
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Compute squared errors and filter outliers based on a threshold
        squared_errors = residuals ** 2
        threshold = np.percentile(squared_errors, 99)  # Remove the top 5% largest errors
        df = df[squared_errors < threshold]

    return df

def main(input_file, output_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Remove outliers
    df_no_outliers = remove_outliers(df)

    # Save the cleaned data
    df_no_outliers.to_parquet(output_file, index=False)
    print(f"Outliers removed. Cleaned data saved to {output_file}.")

if __name__ == "__main__":
    file_path = sys.argv[1]
    main(file_path, file_path)

