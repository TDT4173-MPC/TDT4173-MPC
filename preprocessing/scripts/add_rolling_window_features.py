import pandas as pd
import sys

def add_rolling_window_features(df, features_with_rolling_windows):
    """
    Adds rolling window average columns for specified features in the dataframe with specific window sizes.
    'features_with_rolling_windows' is a dictionary with features as keys and window sizes as values.
    """
    for feature, window_size in features_with_rolling_windows.items():
        rolling_column_name = f"{feature}_rolling_avg_{window_size}"
        df[rolling_column_name] = df[feature].rolling(window=window_size).mean().fillna(df[feature])
    return df

def main(input_file):
    
    # Hardcode the dictionary of features and their rolling window sizes
    features_with_rolling_windows = {
        't_1000hPa:K': 24,  
        'msl_pressure:hPa': 24,
        'absolute_humidity_2m:gm3': 24,
        'total_cloud_cover:p': 6,
        'sun_elevation:d': 6,
        'total_radiation': 24,
        'total_radiation': 3
    }


    # Read the data
    df = pd.read_parquet(input_file)
    
    # Add rolling window features
    df_with_rolling_windows = add_rolling_window_features(df, features_with_rolling_windows)

    # Save the modified data back to the same path
    df_with_rolling_windows.to_parquet(input_file, index=False)
    print(f"Rolling window features added for specified features and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
