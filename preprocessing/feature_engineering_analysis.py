import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import data
obs_A = pd.read_parquet('preprocessing/data/obs_A.parquet')
est_A = pd.read_parquet('preprocessing/data/est_A.parquet')
obs_B = pd.read_parquet('preprocessing/data/obs_B.parquet')
est_B = pd.read_parquet('preprocessing/data/est_B.parquet')
obs_C = pd.read_parquet('preprocessing/data/obs_C.parquet')
est_C = pd.read_parquet('preprocessing/data/est_C.parquet')
test_A = pd.read_parquet('preprocessing/data/test_A.parquet')
test_B = pd.read_parquet('preprocessing/data/test_B.parquet')
test_C = pd.read_parquet('preprocessing/data/test_C.parquet')

# Concatenate data
A = pd.concat([obs_A, est_A], axis=1)
B = pd.concat([obs_B, est_B], axis=1)
C = pd.concat([obs_C, est_C], axis=1)

# List of features to iterate over
features = [
    "diffuse_rad:W", "direct_rad:W",
    "effective_cloud_cover:p", "fresh_snow_24h:cm", "sun_elevation:d",
    "absolute_humidity_2m:gm3", "super_cooled_liquid_water:kgm2",
    "t_1000hPa:K", "total_cloud_cover:p", "air_density_2m:kgm3",
    "clear_sky_rad:W", "visibility:m", "relative_humidity_1000hPa:p",
    "msl_pressure:hPa", "snow_water:kgm2", "dew_point_2m:K",
    "wind_speed_u_10m:ms", "direct_rad_1h:J", "diffuse_rad_1h:J",
    "clear_sky_energy_1h:J", "wind_speed_10m:ms", "wind_speed_v_10m:ms",
    "elevation:m", "date_calc", "precip_5min:mm", "is_day:idx",
    "is_in_shadow:idx", "precip_type_5min:idx", "pressure_100m:hPa",
    "pressure_50m:hPa", "rain_water:kgm2", "sfc_pressure:hPa",
    "snow_depth:cm", "snow_melt_10min:mm", "sun_azimuth:d",
    "ceiling_height_agl:m", "cloud_base_agl:m", "prob_rime:p",
    "dew_or_rime:idx", "fresh_snow_3h:cm", "snow_density:kgm3",
    "fresh_snow_6h:cm", "fresh_snow_12h:cm", "fresh_snow_1h:cm",
    "wind_speed_w_1000hPa:ms", "snow_drift:idx", "month", "year", "hour"
]


# Splitting the dataset
X = A.drop('pv_measurement', axis=1)
y = A['pv_measurement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DataFrame to store results
results = pd.DataFrame(columns=['Feature', 'MSE_Original', 'MAE_Original', 'MSE_Squared', 'MAE_Squared'])

for feature in features:

    # Adding squared feature
    X_train_squared = X_train.copy()
    X_test_squared = X_test.copy()
    print(feature)
    print(X_train[feature])
    print(X_train.columns)
    X_train_squared[feature + '_squared'] = X_train[feature] ** 2
    X_test_squared[feature + '_squared'] = X_test[feature] ** 2
    
    # Training model with original feature
    model_original = RandomForestRegressor(random_state=42)
    model_original.fit(X_train[[feature]], y_train)
    y_pred_original = model_original.predict(X_test[[feature]])
    mse_original = mean_squared_error(y_test, y_pred_original)
    mae_original = mean_absolute_error(y_test, y_pred_original)
    
    # Training model with squared feature
    model_squared = RandomForestRegressor(random_state=42)
    model_squared.fit(X_train_squared[[feature, feature + '_squared']], y_train)
    y_pred_squared = model_squared.predict(X_test_squared[[feature, feature + '_squared']])
    mse_squared = mean_squared_error(y_test, y_pred_squared)
    mae_squared = mean_absolute_error(y_test, y_pred_squared)
    
    # Saving results
    results = results.append({
        'Feature': feature,
        'MSE_Original': mse_original,
        'MAE_Original': mae_original,
        'MSE_Squared': mse_squared,
        'MAE_Squared': mae_squared
    }, ignore_index=True)

# Display the results
print(results)
