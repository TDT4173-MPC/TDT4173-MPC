import pandas as pd

def main():

    # Read in the data
    x_target_A = pd.read_parquet('./data/A/train_targets.parquet')
    x_train_obs_A = pd.read_parquet('./data/A/X_train_observed.parquet')
    x_train_est_A = pd.read_parquet('./data/A/X_train_estimated.parquet')

    x_target_B = pd.read_parquet('./data/B/train_targets.parquet')
    x_train_obs_B = pd.read_parquet('./data/B/X_train_observed.parquet')
    x_train_est_B = pd.read_parquet('./data/B/X_train_estimated.parquet')

    x_target_C = pd.read_parquet('./data/C/train_targets.parquet')
    x_train_obs_C = pd.read_parquet('./data/C/X_train_observed.parquet')
    x_train_est_C = pd.read_parquet('./data/C/X_train_estimated.parquet')

    # Resample
    x_train_obs_A_resampled = x_train_obs_A.set_index('date_forecast').resample('1H').mean()
    x_train_est_A_resampled = x_train_est_A.set_index('date_forecast').resample('1H').mean()

    x_train_obs_B_resampled = x_train_obs_B.set_index('date_forecast').resample('1H').mean()
    x_train_est_B_resampled = x_train_est_B.set_index('date_forecast').resample('1H').mean()

    x_train_obs_C_resampled = x_train_obs_C.set_index('date_forecast').resample('1H').mean()
    x_train_est_C_resampled = x_train_est_C.set_index('date_forecast').resample('1H').mean()

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

    # Save
    obs_A.to_csv('preprocessing/data/obs_A.csv', index=False)
    est_A.to_csv('preprocessing/data/est_A.csv', index=False)

    obs_B.to_csv('preprocessing/data/obs_B.csv' , index=False)
    est_B.to_csv('preprocessing/data/est_B.csv', index=False)

    obs_C.to_csv('preprocessing/data/obs_C.csv', index=False)
    est_C.to_csv('preprocessing/data/est_C.csv', index=False)


if __name__ == '__main__':
    main()
    # Print the filenames
    files = ["obs_A.csv", "est_A.csv", "obs_B.csv", "est_B.csv", "obs_C.csv", "est_C.csv"]
    for file in files:
        print(file)