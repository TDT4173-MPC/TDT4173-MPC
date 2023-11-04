import pandas as pd
import numpy as np
import pywt
import sys

def add_wavelet_features(df, features):
    """
    Adds Wavelet transform features for specified features in the dataframe.
    """
    for feature in features:
        data = df[feature].values
        (cA, cD) = pywt.dwt(data, 'db1')  # 'db1' denotes Daubechies wavelet with one vanishing moment
        df[feature + '_wavelet_approx'] = np.pad(cA, (0, len(data) - len(cA)), 'constant', constant_values=(0,))
        df[feature + '_wavelet_detail'] = np.pad(cD, (0, len(data) - len(cD)), 'constant', constant_values=(0,))
    return df

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Define the features for which to calculate the Wavelet transform
    # Since you want to perform it on all columns, we fetch all column names
    features_to_wavelet = df.columns.tolist()

    # Remove any non-numeric or previously generated transform columns to avoid redundancy
    features_to_wavelet = [f for f in features_to_wavelet if not f.endswith(('_fft_amplitude', '_fft_phase'))]

    # Add Wavelet transform features
    df_with_wavelets = add_wavelet_features(df, features_to_wavelet)

    # Save the modified data back to the same path
    df_with_wavelets.to_parquet(input_file, index=False)
    print(f"Wavelet transform features added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
