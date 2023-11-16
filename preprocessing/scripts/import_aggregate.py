import pandas as pd


def main():

    # Columns that messes up test data
    columns = [
        
         ]
    
    aggregation_methods = {
        'date_forecast': 'first',
        'diffuse_rad:W': 'sum',
        'direct_rad:W': 'last',
        'clear_sky_rad:W': 'sum',
        'diffuse_rad_1h:J': 'last',
        'direct_rad_1h:J': 'last',
        'clear_sky_energy_1h:J': 'last',
        'absolute_humidity_2m:gm3': 'mean',
        'air_density_2m:kgm3': 'mean',
        'ceiling_height_agl:m': 'max',
        'cloud_base_agl:m': 'mean',
        'dew_or_rime:idx': 'min',
        'dew_point_2m:K': 'mean',
        'effective_cloud_cover:p': 'sum',
        'elevation:m': 'first',
        'fresh_snow_12h:cm': 'max',
        'fresh_snow_1h:cm': 'sum',
        'fresh_snow_24h:cm': 'max',
        'fresh_snow_3h:cm': 'max',
        'fresh_snow_6h:cm': 'max',
        'is_day:idx': 'max',
        'is_in_shadow:idx': 'max',
        'msl_pressure:hPa': 'mean',
        'precip_5min:mm': 'sum',
        'precip_type_5min:idx': 'sum',
        'pressure_100m:hPa': 'mean',
        'pressure_50m:hPa': 'mean',
        'prob_rime:p': 'max',
        'rain_water:kgm2': 'sum',
        'relative_humidity_1000hPa:p': 'mean',
        'sfc_pressure:hPa': 'mean',
        'snow_density:kgm3': 'mean',
        'snow_depth:cm': 'max',
        'snow_drift:idx': 'max',
        'snow_melt_10min:mm': 'sum',
        'snow_water:kgm2': 'sum',
        'sun_azimuth:d': 'first',
        'sun_elevation:d': 'sum',
        'super_cooled_liquid_water:kgm2': 'sum',
        't_1000hPa:K': 'mean',
        'total_cloud_cover:p': 'mean',
        'visibility:m': 'mean',
        'wind_speed_10m:ms': 'mean',
        'wind_speed_u_10m:ms': 'mean',
        'wind_speed_v_10m:ms': 'mean',
        'wind_speed_w_1000hPa:ms': 'mean',
        'cloud_base_agl:m': 'max',
        'snow_density:kgm3': 'mean'
    }


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

    # Fix cloud data for test set
    x_test_est_A['effective_cloud_cover:p'] = x_test_est_A['effective_cloud_cover:p'].fillna(0)
    x_test_est_B['effective_cloud_cover:p'] = x_test_est_B['effective_cloud_cover:p'].fillna(0)
    x_test_est_C['effective_cloud_cover:p'] = x_test_est_C['effective_cloud_cover:p'].fillna(0)

    x_test_est_A['total_cloud_cover:p'] = x_test_est_A['total_cloud_cover:p'].fillna(0)
    x_test_est_B['total_cloud_cover:p'] = x_test_est_B['total_cloud_cover:p'].fillna(0)
    x_test_est_C['total_cloud_cover:p'] = x_test_est_C['total_cloud_cover:p'].fillna(0)

    x_test_est_A['cloud_base_agl:m'] = x_test_est_A['cloud_base_agl:m'].fillna(0)
    x_test_est_B['cloud_base_agl:m'] = x_test_est_B['cloud_base_agl:m'].fillna(0)
    x_test_est_C['cloud_base_agl:m'] = x_test_est_C['cloud_base_agl:m'].fillna(0)

    x_test_est_A['ceiling_height_agl:m'] = x_test_est_A['ceiling_height_agl:m'].fillna(0)
    x_test_est_B['ceiling_height_agl:m'] = x_test_est_B['ceiling_height_agl:m'].fillna(0)
    x_test_est_C['ceiling_height_agl:m'] = x_test_est_C['ceiling_height_agl:m'].fillna(0)

    x_test_est_A['snow_density:kgm3'] = x_test_est_A['snow_density:kgm3'].fillna(0)
    x_test_est_B['snow_density:kgm3'] = x_test_est_B['snow_density:kgm3'].fillna(0)
    x_test_est_C['snow_density:kgm3'] = x_test_est_C['snow_density:kgm3'].fillna(0)

    x_test_est_A['snow_drift:idx'] = x_test_est_A['snow_drift:idx'].fillna(0)
    x_test_est_B['snow_drift:idx'] = x_test_est_B['snow_drift:idx'].fillna(0)
    x_test_est_C['snow_drift:idx'] = x_test_est_C['snow_drift:idx'].fillna(0)

    # Resample
    x_train_obs_A_resampled = x_train_obs_A.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)
    x_train_est_A_resampled = x_train_est_A.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)
    x_test_est_A_resampled = x_test_est_A.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)

    x_train_obs_B_resampled = x_train_obs_B.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)
    x_train_est_B_resampled = x_train_est_B.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)
    x_test_est_B_resampled = x_test_est_B.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)

    x_train_obs_C_resampled = x_train_obs_C.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)
    x_train_est_C_resampled = x_train_est_C.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)
    x_test_est_C_resampled = x_test_est_C.groupby(pd.Grouper(key='date_forecast', freq='1H')).aggregate(aggregation_methods)

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
    test_A = x_test_est_A_resampled
    test_B = x_test_est_B_resampled
    test_C = x_test_est_C_resampled

    # Clean up test data
    test_A = test_A.drop(columns=columns)
    test_B = test_B.drop(columns=columns)
    test_C = test_C.drop(columns=columns)

    test_A = test_A.dropna()
    test_B = test_B.dropna()
    test_C = test_C.dropna()

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