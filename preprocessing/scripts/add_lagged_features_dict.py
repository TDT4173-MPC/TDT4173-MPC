import pandas as pd
import sys

def add_lagged_features(df, features_with__lags, fill_value=None):
    """
    Adds lagged columns for specified features in the dataframe with specific lag periods.
    'features_with_specific_lags' is a dictionary with features as keys and specific lag as values.
    'fill_value' is what to fill the NaNs with, after shifting.
    """
    for feature, specific_lag in features_with__lags.items():
        lag_column_name = f"{feature}_lag_{specific_lag}"
        df[lag_column_name] = df[feature].shift(specific_lag).fillna(fill_value)
    return df

def main(input_file):
    # Hardcode the dictionary of features and their specific lags

    features_with_lags = {
        "sun_azimuth:d": 8,
    }

    # Read the data
    df = pd.read_parquet(input_file)

    # Add lagged features for specific lags
    df_with_specific_lags = add_lagged_features(df, features_with_lags, fill_value=0) # You can change the fill_value as needed

    # Save the modified data back to the same path
    df_with_specific_lags.to_parquet(input_file, index=False)
    print(f"Specific lagged features added for specified features and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)



