import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Filter out FutureWarnings from scikit-learn library
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import data
obs_A = pd.read_parquet('preprocessing/data/obs_A.parquet').drop(columns='date_forecast')
est_A = pd.read_parquet('preprocessing/data/est_A.parquet').drop(columns='date_forecast')
obs_B = pd.read_parquet('preprocessing/data/obs_B.parquet').drop(columns='date_forecast')
est_B = pd.read_parquet('preprocessing/data/est_B.parquet').drop(columns='date_forecast')
obs_C = pd.read_parquet('preprocessing/data/obs_C.parquet').drop(columns='date_forecast')
est_C = pd.read_parquet('preprocessing/data/est_C.parquet').drop(columns='date_forecast')
test_A = pd.read_parquet('preprocessing/data/test_A.parquet').drop(columns='date_forecast')
test_B = pd.read_parquet('preprocessing/data/test_B.parquet').drop(columns='date_forecast')
test_C = pd.read_parquet('preprocessing/data/test_C.parquet').drop(columns='date_forecast')

# Concatenate data
A = pd.concat([obs_A.add_suffix('_obs'), est_A.add_suffix('_est')], axis=1)
B = pd.concat([obs_B.add_suffix('_obs'), est_B.add_suffix('_est')], axis=1)
C = pd.concat([obs_C.add_suffix('_obs'), est_C.add_suffix('_est')], axis=1)

# Configurations
n_estimators = 200
tree_depth = 15

# List of features to iterate over
features = [
    # "diffuse_rad:W", "direct_rad:W",
    # "effective_cloud_cover:p", "fresh_snow_24h:cm"
    # , "sun_elevation:d",
    # "absolute_humidity_2m:gm3", "super_cooled_liquid_water:kgm2",
    # "t_1000hPa:K", "total_cloud_cover:p", "air_density_2m:kgm3",
    # "clear_sky_rad:W", "visibility:m", "relative_humidity_1000hPa:p",
    # "msl_pressure:hPa", "snow_water:kgm2", 
    # "dew_point_2m:K",
    # "wind_speed_u_10m:ms", "direct_rad_1h:J", "diffuse_rad_1h:J",
    # "clear_sky_energy_1h:J", "wind_speed_10m:ms", "wind_speed_v_10m:ms",
    # "elevation:m", "precip_5min:mm", "is_day:idx",
    # "is_in_shadow:idx", "precip_type_5min:idx", "pressure_100m:hPa",
    # "pressure_50m:hPa", "rain_water:kgm2", "sfc_pressure:hPa",
    # "snow_depth:cm", "snow_melt_10min:mm", "sun_azimuth:d",
    # "ceiling_height_agl:m", "cloud_base_agl:m", "prob_rime:p",
    # "dew_or_rime:idx", "fresh_snow_3h:cm", "snow_density:kgm3",
    # "fresh_snow_6h:cm", "fresh_snow_12h:cm", "fresh_snow_1h:cm",
    # "wind_speed_w_1000hPa:ms", "snow_drift:idx", "month", "year", "hour"
    'temp_dew_point_diff:K'
]


# Splitting the dataset
X = obs_A.drop('pv_measurement', axis=1)
y = obs_A['pv_measurement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DataFrame to store results
results = pd.DataFrame(columns=['Feature',  'MAE_Original', 'MAE_Squared', 
                                'MAE_Cubed', 'MAE_SquareRoot',
                                'MAE_Normalized', 'MAE_Standardized', 'MAE_Integral',
                                'Importance_Original', 'Importance_Squared',
                                'Importance_Cubed', 'Importance_SquareRoot',
                                'Importance_Normalized', 'Importance_Standardized', 'Importance_Integral',
                                'MSE_Original', 'MSE_Squared',
                                'MSE_Cubed',  'MSE_SquareRoot', 
                                'MSE_Normalized', 'MSE_Standardized', 'MSE_Integral'])

# Training model with original features
model_original = RandomForestRegressor(random_state=42, n_estimators=n_estimators,  max_depth=tree_depth)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)
mae_original = mean_absolute_error(y_test, y_pred_original)

# Extracing feature importances
feature_importances = model_original.feature_importances_



for feature in features:

    try:
        # Saving results
        new_index = len(results)
        results.loc[new_index] = [np.nan] * len(results.columns)
        results.at[new_index, 'Feature'] = feature
        feature_index = list(X_train.columns).index(feature)
        results.at[new_index, 'Importance_Original'] = feature_importances[feature_index]
        results.at[new_index, 'MSE_Original'] = mse_original
        results.at[new_index, 'MAE_Original'] = mae_original
    except Exception as e:
        print(f"An error occurred while processing {feature}_original: {e}")

    try:
        # Adding squared feature
        X_train_squared = X_train.copy()
        X_test_squared = X_test.copy()
        X_train_squared[feature + '_squared'] = X_train[feature] ** 2
        X_test_squared[feature + '_squared'] = X_test[feature] ** 2
        
        # Training model with squared feature
        model_squared = RandomForestRegressor(random_state=42, n_estimators=n_estimators,  max_depth=tree_depth)
        model_squared.fit(X_train_squared, y_train)
        y_pred_squared = model_squared.predict(X_test_squared)
        mse_squared = mean_squared_error(y_test, y_pred_squared)
        mae_squared = mean_absolute_error(y_test, y_pred_squared)

        # Extracting feature importances
        feature_importances = model_squared.feature_importances_
        feature_index = list(X_train_squared.columns).index(feature + '_squared')
        importance_of_feature = feature_importances[feature_index]
        
        # Saving results
        results.at[new_index, 'Importance_Squared'] = importance_of_feature
        results.at[new_index, 'MSE_Squared'] = mse_squared
        results.at[new_index, 'MAE_Squared'] = mae_squared

        print(f'{feature}_squared done')
    except Exception as e:
        print(f"An error occurred while processing {feature}_squared: {e}")


    try:
        # Adding cubed feature
        X_train_cubed = X_train.copy()
        X_test_cubed = X_test.copy()
        X_train_cubed[feature + '_cubed'] = X_train[feature] ** 3
        X_test_cubed[feature + '_cubed'] = X_test[feature] ** 3
        
        # Training model with cubed feature
        model_cubed = RandomForestRegressor(random_state=42, n_estimators=n_estimators,  max_depth=tree_depth)
        model_cubed.fit(X_train_cubed, y_train)
        y_pred_cubed = model_cubed.predict(X_test_cubed)
        mse_cubed = mean_squared_error(y_test, y_pred_cubed)
        mae_cubed = mean_absolute_error(y_test, y_pred_cubed)

        # Extract feature importances
        feature_importances = model_cubed.feature_importances_
        feature_index = list(X_train_cubed.columns).index(feature + '_cubed')
        importance_of_feature = feature_importances[feature_index]
        
        # Saving results
        results.at[new_index, 'Importance_Cubed'] = importance_of_feature
        results.at[new_index, 'MSE_Cubed'] = mse_cubed
        results.at[new_index, 'MAE_Cubed'] = mae_cubed

        print(f'{feature}_cubed done')
    except Exception as e:
        print(f"An error occurred while processing {feature}_cubed: {e}")


    try:
        # Adding square root feature
        X_train_square_root = X_train.copy()
        X_test_square_root = X_test.copy()
        X_train_square_root[feature + '_square_root'] = X_train[feature] ** 0.5
        X_test_square_root[feature + '_square_root'] = X_test[feature] ** 0.5

        # Training model with square root feature
        model_square_root = RandomForestRegressor(random_state=42, n_estimators=n_estimators,  max_depth=tree_depth)
        model_square_root.fit(X_train_square_root, y_train)
        y_pred_square_root = model_square_root.predict(X_test_square_root)
        mse_square_root = mean_squared_error(y_test, y_pred_square_root)
        mae_square_root = mean_absolute_error(y_test, y_pred_square_root)

        # Extract feature importances
        feature_importances = model_square_root.feature_importances_
        feature_index = list(X_train_square_root.columns).index(feature + '_square_root')
        importance_of_feature = feature_importances[feature_index]

        # Saving results
        results.at[new_index, 'Importance_SquareRoot'] = importance_of_feature
        results.at[new_index, 'MSE_SquareRoot'] = mse_square_root
        results.at[new_index, 'MAE_SquareRoot'] = mae_square_root

        print(f'{feature}_square_root done')
    except Exception as e:
        print(f"An error occurred while processing {feature}_square_root: {e}")


    try:
        # Adding normalized feature
        X_train_normalized = X_train.copy()
        X_test_normalized = X_test.copy()
        X_train_normalized[feature + '_normalized'] = (X_train[feature] - X_train[feature].mean()) / X_train[feature].std()
        X_test_normalized[feature + '_normalized'] = (X_test[feature] - X_test[feature].mean()) / X_test[feature].std()

        # Training model with normalized feature
        model_normalized = RandomForestRegressor(random_state=42, n_estimators=n_estimators,  max_depth=tree_depth)
        model_normalized.fit(X_train_normalized, y_train)
        y_pred_normalized = model_normalized.predict(X_test_normalized)
        mse_normalized = mean_squared_error(y_test, y_pred_normalized)
        mae_normalized = mean_absolute_error(y_test, y_pred_normalized)

        # Extract feature importances
        feature_importances = model_normalized.feature_importances_
        feature_index = list(X_train_normalized.columns).index(feature + '_normalized')
        importance_of_feature = feature_importances[feature_index]

        # Saving results
        results.at[new_index, 'Importance_Normalized'] = importance_of_feature
        results.at[new_index, 'MSE_Normalized'] = mse_normalized
        results.at[new_index, 'MAE_Normalized'] = mae_normalized

        print(f'{feature}_normalized done')
    except Exception as e:
        print(f"An error occurred while processing {feature}_normalized: {e}")


    try:
        # Adding standardized feature
        X_train_standardized = X_train.copy()
        X_test_standardized = X_test.copy()
        X_train_standardized[feature + '_standardized'] = (X_train[feature] - X_train[feature].mean()) / X_train[feature].std()
        X_test_standardized[feature + '_standardized'] = (X_test[feature] - X_test[feature].mean()) / X_test[feature].std()

        # Training model with standardized feature
        model_standardized = RandomForestRegressor(random_state=42, n_estimators=n_estimators,  max_depth=tree_depth)
        model_standardized.fit(X_train_standardized, y_train)
        y_pred_standardized = model_standardized.predict(X_test_standardized)
        mse_standardized = mean_squared_error(y_test, y_pred_standardized)
        mae_standardized = mean_absolute_error(y_test, y_pred_standardized)

        # Extract feature importances
        feature_importances = model_standardized.feature_importances_
        feature_index = list(X_train_standardized.columns).index(feature + '_standardized')
        importance_of_feature = feature_importances[feature_index]

        # Saving results
        results.at[new_index, 'Importance_Standardized'] = importance_of_feature
        results.at[new_index, 'MSE_Standardized'] = mse_standardized
        results.at[new_index, 'MAE_Standardized'] = mae_standardized

        print(f'{feature}_standardized done')
    except Exception as e:
        print(f"An error occurred while processing {feature}_standardized: {e}")

    
    try:
        # Adding integral effect feature with a rolling window of 3
        X_train_integral = X_train.copy()
        X_test_integral = X_test.copy()
        X_train_integral[feature + '_integral'] = X_train[feature].rolling(window=3).sum()
        X_test_integral[feature + '_integral'] = X_test[feature].rolling(window=3).sum()
        X_train_integral[feature + '_integral'] = X_train_integral[feature + '_integral'].fillna(0)
        X_test_integral[feature + '_integral'] = X_test_integral[feature + '_integral'].fillna(0)


        # Training model with integral feature
        model_integral = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_depth=tree_depth)
        model_integral.fit(X_train_integral, y_train)
        y_pred_integral = model_integral.predict(X_test_integral)
        mse_integral = mean_squared_error(y_test, y_pred_integral)
        mae_integral = mean_absolute_error(y_test, y_pred_integral)

        # Extract feature importances
        feature_importances = model_integral.feature_importances_
        feature_index = list(X_train_integral.columns).index(feature + '_integral')
        importance_of_feature = feature_importances[feature_index]

        # Saving results
        results.at[new_index, 'Importance_Integral'] = importance_of_feature
        results.at[new_index, 'MSE_Integral'] = mse_integral
        results.at[new_index, 'MAE_Integral'] = mae_integral

        print(f'{feature}_integral done')
    except Exception as e:
        print(f"An error occurred while processing {feature}_integral: {e}")



# Display the results
print(results)

# Save the results
results.to_csv('feature_engineering_analysis.csv', index=False)
print('Results saved to feature_engineering_analysis.csv')

