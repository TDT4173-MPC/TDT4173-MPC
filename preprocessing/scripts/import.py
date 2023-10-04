import pandas as pd

def add_time_features(df):
    """
    Adds a 'calc_time_before' column to the dataframe.
    If the dataframe has a 'date_calc' column, the new column will contain the difference 
    in hours between 'date_forecast' and 'date_calc'. 
    If there's no 'date_calc' column, the new column will contain all zeros.
    """
    
    # Check if 'date_calc' exists in the dataframe
    if 'date_calc' in df.columns:
        # Convert the columns to datetime format
        df['date_forecast'] = pd.to_datetime(df['date_forecast'])
        df['date_calc'] = pd.to_datetime(df['date_calc'])
        
        # Calculate the difference in hours
        df['calc_time_before'] = (df['date_forecast'] - df['date_calc']).dt.total_seconds() / 3600
    else:
        # If 'date_calc' doesn't exist, add 'calc_time_before' with all zeros
        df['calc_time_before'] = 0

    return df


def main():

    # Read in the data
    x_target_A = pd.read_parquet('./data/A/train_targets.parquet')
    x_train_obs_A = pd.read_parquet('./data/A/X_train_observed.parquet')
    x_train_est_A = pd.read_parquet('./data/A/X_train_estimated.parquet')
    x_test_est_A = pd.read_parquet('./data/A/X_test_estimated.parquet')

    x_target_B = pd.read_parquet('./data/B/train_targets.parquet')
    x_train_obs_B = pd.read_parquet('./data/B/X_train_observed.parquet')
    x_train_est_B = pd.read_parquet('./data/B/X_train_estimated.parquet')
    x_test_est_B = pd.read_parquet('./data/B/X_test_estimated.parquet')

    x_target_C = pd.read_parquet('./data/C/train_targets.parquet')
    x_train_obs_C = pd.read_parquet('./data/C/X_train_observed.parquet')
    x_train_est_C = pd.read_parquet('./data/C/X_train_estimated.parquet')
    x_test_est_C = pd.read_parquet('./data/C/X_test_estimated.parquet')

    # Add time features
    # x_train_obs_A = add_time_features(x_train_obs_A)
    # x_train_est_A = add_time_features(x_train_est_A)
    # x_test_est_A = add_time_features(x_test_est_A)

    # x_train_obs_B = add_time_features(x_train_obs_B)
    # x_train_est_B = add_time_features(x_train_est_B)
    # x_test_est_B = add_time_features(x_test_est_B)

    # x_train_obs_C = add_time_features(x_train_obs_C)
    # x_train_est_C = add_time_features(x_train_est_C)
    # x_test_est_C = add_time_features(x_test_est_C)

    # Resample
    x_train_obs_A_resampled = x_train_obs_A.set_index('date_forecast').resample('1H').mean()
    x_train_est_A_resampled = x_train_est_A.set_index('date_forecast').resample('1H').mean()
    x_test_est_A_resampled = x_test_est_A.set_index('date_forecast').resample('1H').mean()

    x_train_obs_B_resampled = x_train_obs_B.set_index('date_forecast').resample('1H').mean()
    x_train_est_B_resampled = x_train_est_B.set_index('date_forecast').resample('1H').mean()
    x_test_est_B_resampled = x_test_est_B.set_index('date_forecast').resample('1H').mean()

    x_train_obs_C_resampled = x_train_obs_C.set_index('date_forecast').resample('1H').mean()
    x_train_est_C_resampled = x_train_est_C.set_index('date_forecast').resample('1H').mean()
    x_test_est_C_resampled = x_test_est_C.set_index('date_forecast').resample('1H').mean()

    # Merge
    split_value = x_train_est_A['date_forecast'].iloc[0]
    split_index = x_target_A[x_target_A['time'] == split_value].index[0]

    x_target_obs_A = x_target_A.iloc[:split_index]
    x_target_est_A = x_target_A.iloc[split_index:]

    obs_A = x_train_obs_A_resampled.merge(x_target_obs_A, left_index=True, right_on='time')
    est_A = x_train_est_A_resampled.merge(x_target_est_A, left_index=True, right_on='time')

    split_value = x_train_est_B['date_forecast'].iloc[0]
    split_index = x_target_B[x_target_B['time'] == split_value].index[0]

    x_target_obs_B = x_target_B.iloc[:split_index]
    x_target_est_B = x_target_B.iloc[split_index:]

    obs_B = x_train_obs_B_resampled.merge(x_target_obs_B, left_index=True, right_on='time')
    est_B = x_train_est_B_resampled.merge(x_target_est_B, left_index=True, right_on='time')

    split_value = x_train_est_C['date_forecast'].iloc[0]
    split_index = x_target_C[x_target_C['time'] == split_value].index[0]

    x_target_obs_C = x_target_C.iloc[:split_index]
    x_target_est_C = x_target_C.iloc[split_index:]

    obs_C = x_train_obs_C_resampled.merge(x_target_obs_C, left_index=True, right_on='time')
    est_C = x_train_est_C_resampled.merge(x_target_est_C, left_index=True, right_on='time')

    # Drop NaNs in test set
    test_A = x_test_est_A_resampled.dropna()
    test_B = x_test_est_B_resampled.dropna()
    test_C = x_test_est_C_resampled.dropna()

    # Save
    obs_A.to_parquet('preprocessing/data/obs_A.parquet', index=False)
    est_A.to_parquet('preprocessing/data/est_A.parquet', index=False)
    test_A.to_parquet('preprocessing/data/test_A.parquet', index=False)

    obs_B.to_parquet('preprocessing/data/obs_B.parquet' , index=False)
    est_B.to_parquet('preprocessing/data/est_B.parquet', index=False)
    test_B.to_parquet('preprocessing/data/test_B.parquet', index=False)

    obs_C.to_parquet('preprocessing/data/obs_C.parquet', index=False)
    est_C.to_parquet('preprocessing/data/est_C.parquet', index=False)
    test_C.to_parquet('preprocessing/data/test_C.parquet', index=False)


if __name__ == '__main__':
    main()
    # Print the filenames
    files = ["obs_A.parquet", "est_A.parquet", "test_A.parquet", "obs_B.parquet", "est_B.parquet", "test_B.parquet", "obs_C.parquet", "est_C.parquet", "test_C.parquet"]
    for file in files:
        print(file)