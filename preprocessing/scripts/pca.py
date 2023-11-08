import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)
    
    # Standardize the data
    scaler = StandardScaler()
    try:
        popped_cols = df['pv_measurement', 'date_forecast']
        features_standardized = scaler.fit_transform(df.drop(columns=['pv_measurement', 'date_forecast']))
    except KeyError:
        popped_cols = df['date_forecast']
        features_standardized = scaler.fit_transform(df.drop(columns=['date_forecast']))

    # Apply PCA
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(features_standardized)

    # Create a DataFrame with the PCA results
    df_pca = pd.DataFrame(data=pca_result)

    # If you want to keep non-numeric columns and concatenate them back to the PCA result:
    try:
        df_final = pd.concat([popped_cols, df_pca], axis=1)
    except:
        df_final = df_pca
    
    # Save the modified data back to the same path
    df_final.to_parquet(input_file, index=False)
    print(f"PCA applied, and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
