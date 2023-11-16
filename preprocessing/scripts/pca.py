import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Separate the 'pv_measurement' column
    try:
        pv_measurement = df[['pv_measurement']]
        df = df.drop(columns=['pv_measurement'])
    except KeyError:
        print("Column 'pv_measurement' not found. Applying PCA to all columns.")
        pv_measurement = None

    # Standardize the remaining data
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(df)

    # Apply PCA
    pca = PCA(n_components=40)
    pca_result = pca.fit_transform(features_standardized)

    # Create a DataFrame with the PCA results
    df_pca = pd.DataFrame(data=pca_result, columns=[f'PCA_{i}' for i in range(pca.n_components_)])

    # Concatenate the 'pv_measurement' column back if it exists
    if pv_measurement is not None:
        df_final = pd.concat([pv_measurement, df_pca], axis=1)
    else:
        df_final = df_pca
    
    # Save the modified data back to the same path
    df_final.to_parquet(input_file, index=False)
    print(f"PCA applied, and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
