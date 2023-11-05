import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

def add_fourier_features(df, features):
    """
    Adds Fourier transform features for specified features in the dataframe.
    """
    for feature in features:
        # Check if the feature exists and has non-NaN values
        if feature in df.columns and df[feature].notna().any():
            # Compute the fast Fourier transform
            fft_values = np.fft.fft(df[feature].dropna().values)  # Drop NaN values
            
            # Compute the absolute values to get amplitudes
            fft_abs = np.abs(fft_values)
            
            # Compute the angles to get phase information
            fft_phase = np.angle(fft_values)
            
            # Add the amplitudes and phases to the dataframe
            df[feature + '_fft_amplitude'] = np.nan  # Initialize with NaN
            df[feature + '_fft_phase'] = np.nan  # Initialize with NaN
            
            # Assign values only to the non-NaN indices
            non_nan_indices = df.index[df[feature].notna()]
            df.loc[non_nan_indices, feature + '_fft_amplitude'] = fft_abs
            df.loc[non_nan_indices, feature + '_fft_phase'] = fft_phase
        else:
            print(f"Warning: '{feature}' does not exist or contains no non-NaN values.")
    return df


def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Define the features for which to calculate the Fourier transform
    features_to_fft = ['date_forecast', 'sun_elevation:d']

    # Add Fourier transform features
    df_with_fft = add_fourier_features(df, features_to_fft)

    # Save the modified data back to the same path
    df_with_fft.to_parquet(input_file, index=False)
    print(f"Fourier transform features added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
