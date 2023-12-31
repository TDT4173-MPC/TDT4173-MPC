import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings('ignore')

# Import data
obs_A = pd.read_parquet('./preprocessing/data/obs_A.parquet')
obs_B = pd.read_parquet('./preprocessing/data/obs_B.parquet')
obs_C = pd.read_parquet('./preprocessing/data/obs_C.parquet')
est_A = pd.read_parquet('./preprocessing/data/est_A.parquet')
est_B = pd.read_parquet('./preprocessing/data/est_B.parquet')
est_C = pd.read_parquet('./preprocessing/data/est_C.parquet')
test_A = pd.read_parquet('./preprocessing/data/test_A.parquet')
test_B = pd.read_parquet('./preprocessing/data/test_B.parquet')
test_C = pd.read_parquet('./preprocessing/data/test_C.parquet')

# Columns to drop
# columns = [
#     'date_forecast',
#     'super_cooled_liquid_water:kgm2',
#     'air_density_2m:kgm3',
#     'snow_water:kgm2',
#     'precip_5min:mm',
#     'precip_type_5min:idx',
#     'rain_water:kgm2',
#     'snow_melt_10min:mm',
#     'dew_or_rime:idx',
#     'snow_depth:cm',
#     'prob_rime:p',
#     'is_day:idx',
#     'is_in_shadow:idx',
#     'visibility:m',
#     'relative_humidity_1000hPa:p',
#     'temp_dewpoint_diff'
# ]

# # Drop columns
# obs_A = obs_A.drop(columns=columns)
# obs_B = obs_B.drop(columns=columns)
# obs_C = obs_C.drop(columns=columns)
# est_A = est_A.drop(columns=columns)
# est_B = est_B.drop(columns=columns)
# est_C = est_C.drop(columns=columns)
# test_A = test_A.drop(columns=columns)
# test_B = test_B.drop(columns=columns)
# test_C = test_C.drop(columns=columns)

A = pd.concat([obs_A, est_A])
B = pd.concat([obs_B, est_B])
C = pd.concat([obs_C, est_C])
data = pd.concat([A, B, C])

# Split into X and y
X = data.drop(columns=['pv_measurement'])
y = data['pv_measurement']

params = {  'colsample_bytree': 0.664, 
            'gamma': 3, 
            'learning_rate': 0.012, 
            'max_depth': 12, 
            'min_child_weight': 15, 
            'n_estimators': 300, 
            'reg_alpha': 2, 
            'reg_lambda': 2, 
            'subsample': 0.912 }

# Set up model
model = xgb.XGBRegressor(**params, n_jobs=-1)

# Set up RFE
# sfe = SequentialFeatureSelector(estimator=model, n_features_to_select='auto', n_jobs=-1)
rfe = RFECV(estimator=model, step=1, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1, min_features_to_select=25)

# Fit RFE
# X_transformed = rfe.fit_transform(X, y)

# Save RFE
# pickle.dump(rfe, open('RFECV.pkl', 'wb'))

# Load RFE
rfe = pickle.load(open('RFECV.pkl', 'rb'))

# Get the selected feature indices
selected_features = rfe.get_support(indices=True)

print('Selected features:')
for feature_index in selected_features:
    print(X.columns[feature_index])


"""
Selected features using XGBoost n_estimators=250, max_depth=15:

snow_accumulation \
total_radiation \
sfc_pressure:hPa \
month \
year \
date_forecast_fft_amplitude \
date_forecast_fft_phase \
sun_elevation:d_fft_amplitude \
sun_elevation:d_fft_phase \
clear_sky_rad:W_rate_of_change \
observed \
sun_azimuth:d_lag_7 \
sfc_pressure:hPa_lag_8 \
t_1000hPa:K_lag_4 \
dew_or_rime:idx_lag_11 \
relative_humidity_1000hPa:p_lag_-3 \
temp_dewpoint_diff_lag_-4 \
dew_point_2m:K_lag_19 \
visibility:m_lag_-2 \
t_1000hPa:K_rolling_avg_24 \
msl_pressure:hPa_rolling_avg_24 \
absolute_humidity_2m:gm3_rolling_avg_24 \
total_cloud_cover:p_rolling_avg_6 \
total_radiation_rolling_avg_3 \
diffuse_rad:W \
diffuse_rad_1h:J \
direct_rad:W \
direct_rad_1h:J \
clear_sky_rad:W \
sun_elevation:d \
t_1000hPa:K \
effective_cloud_cover:p \
average_wind_speed \
clear_sky_energy_1h:J \


Selected features using random forest n_estimator=100

clear_sky_energy_1h:J
direct_rad:W
effective_cloud_cover:p
sun_elevation:d
t_1000hPa:K
wind_vector_magnitude
average_wind_speed
total_radiation
sfc_pressure:hPa
month
year
date_forecast_fft_amplitude
date_forecast_fft_phase
sun_elevation:d_fft_amplitude
sun_elevation:d_fft_phase
t_1000hPa:K_rate_of_change
clear_sky_rad:W_rate_of_change
direct_rad:W_rate_of_change
effective_cloud_cover:p_rate_of_change
sun_azimuth:d_lag_7
sfc_pressure:hPa_lag_8
t_1000hPa:K_lag_4
relative_humidity_1000hPa:p_lag_-3
temp_dewpoint_diff_lag_-4
dew_point_2m:K_lag_19
visibility:m_lag_-2
t_1000hPa:K_rolling_avg_24
msl_pressure:hPa_rolling_avg_24
absolute_humidity_2m:gm3_rolling_avg_24
total_cloud_cover:p_rolling_avg_6

"""