import pandas as pd
import sys

def add_rate_of_change_features(df, features, second_order=False):
    """
    Adds rate of change columns for specified features in the dataframe.
    Assumes the dataframe is time sorted. If second_order is True, it also adds the second order rate of change.
    """
    for feature in features:
        rate_column_name = feature + '_rate_of_change'
        df[rate_column_name] = df[feature].diff().fillna(0)  # Handle the first diff NaN if required
        
        if second_order:  # Check if second order difference is required
            second_order_column_name = feature + '_rate_of_change_of_change'
            df[second_order_column_name] = df[rate_column_name].diff().fillna(0)  # Second order difference

    return df

def main(input_file):
    # Define the features for which to calculate rate of change
    features_to_diff = [
        'dew_point_2m:K', 't_1000hPa:K',
        'clear_sky_rad:W', 'diffuse_rad:W', 'direct_rad:W',
        'effective_cloud_cover:p', 'total_cloud_cover:p', 'total_radiation'
    ]

    # Read the data
    df = pd.read_parquet(input_file)

    # Add rate of change features
    df_with_roc = add_rate_of_change_features(df, features_to_diff, second_order=True)

    # Save the modified data back to the same path
    df_with_roc.to_parquet(input_file, index=False)
    print(f"Rate of change and second order features added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
