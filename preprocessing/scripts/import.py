import pandas as pd




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

    # Rename time to date_forecast in target
    x_target_A.rename(columns={'time': 'date_forecast'}, inplace=True)
    x_target_B.rename(columns={'time': 'date_forecast'}, inplace=True)
    x_target_C.rename(columns={'time': 'date_forecast'}, inplace=True)

    # Resample
    x_train_obs_A_resampled = x_train_obs_A.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()
    x_train_est_A_resampled = x_train_est_A.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()
    x_test_est_A_resampled = x_test_est_A.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()

    x_train_obs_B_resampled = x_train_obs_B.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()
    x_train_est_B_resampled = x_train_est_B.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()
    x_test_est_B_resampled = x_test_est_B.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()

    x_train_obs_C_resampled = x_train_obs_C.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()
    x_train_est_C_resampled = x_train_est_C.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()
    x_test_est_C_resampled = x_test_est_C.groupby(pd.Grouper(key='date_forecast', freq='1H')).mean()

    # Merge
    split_value = x_train_est_A['date_forecast'].iloc[0]
    split_index = x_target_A[x_target_A['date_forecast'] == split_value].index[0]

    x_target_obs_A = x_target_A.iloc[:split_index]
    x_target_est_A = x_target_A.iloc[split_index:]

    obs_A = x_train_obs_A_resampled.merge(x_target_obs_A, left_index=True, right_on='date_forecast')
    est_A = x_train_est_A_resampled.merge(x_target_est_A, left_index=True, right_on='date_forecast')

    split_value = x_train_est_B['date_forecast'].iloc[0]
    split_index = x_target_B[x_target_B['date_forecast'] == split_value].index[0]

    x_target_obs_B = x_target_B.iloc[:split_index]
    x_target_est_B = x_target_B.iloc[split_index:]

    obs_B = x_train_obs_B_resampled.merge(x_target_obs_B, left_index=True, right_on='date_forecast')
    est_B = x_train_est_B_resampled.merge(x_target_est_B, left_index=True, right_on='date_forecast')

    split_value = x_train_est_C['date_forecast'].iloc[0]
    split_index = x_target_C[x_target_C['date_forecast'] == split_value].index[0]

    x_target_obs_C = x_target_C.iloc[:split_index]
    x_target_est_C = x_target_C.iloc[split_index:]

    obs_C = x_train_obs_C_resampled.merge(x_target_obs_C, left_index=True, right_on='date_forecast')
    est_C = x_train_est_C_resampled.merge(x_target_est_C, left_index=True, right_on='date_forecast')

    # Keep date_forecast in test dfs
    test_A = x_test_est_A_resampled.reset_index()
    test_B = x_test_est_B_resampled.reset_index()
    test_C = x_test_est_C_resampled.reset_index()

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