import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import train_test_split

def keep_columns(df, columns):
    df = df[columns]
    return df

obs_A = pd.read_csv('./Analysis/Mathias/pipeline/data/obs_A.csv')
est_A = pd.read_csv('./Analysis/Mathias/pipeline/data/est_A.csv')
obs_B = pd.read_csv('./Analysis/Mathias/pipeline/data/obs_B.csv')
est_B = pd.read_csv('./Analysis/Mathias/pipeline/data/est_B.csv')
obs_C = pd.read_csv('./Analysis/Mathias/pipeline/data/obs_C.csv')
est_C = pd.read_csv('./Analysis/Mathias/pipeline/data/est_C.csv')

# Concatenate
A = pd.concat([obs_A, est_A])
B = pd.concat([obs_B, est_B])
C = pd.concat([obs_C, est_C])

# Split into train and test
X_train_A, X_test_A = train_test_split(A, test_size=0.2, shuffle=False)
y_train_A = X_train_A['pv_measurement']
y_test_A = X_test_A['pv_measurement']
X_train_A = X_train_A.drop(columns=['pv_measurement'])
X_test_A = X_test_A.drop(columns=['pv_measurement'])

X_train_B, X_test_B = train_test_split(B, test_size=0.2, shuffle=False)
y_train_B = X_train_B['pv_measurement']
y_test_B = X_test_B['pv_measurement']
X_train_B = X_train_B.drop(columns=['pv_measurement'])
X_test_B = X_test_B.drop(columns=['pv_measurement'])

X_train_C, X_test_C = train_test_split(C, test_size=0.2, shuffle=False)
y_train_C = X_train_C['pv_measurement']
y_test_C = X_test_C['pv_measurement']
X_train_C = X_train_C.drop(columns=['pv_measurement'])
X_test_C = X_test_C.drop(columns=['pv_measurement'])


# Train models

# Random Forest for Location A
rf_A = RandomForestRegressor(n_estimators=10, min_samples_split=2, min_samples_leaf=1, max_features='log2', max_depth=5, bootstrap=True, random_state=0)
rf_A.fit(X_train_A, y_train_A)

# Random Forest for Location B
rf_B = RandomForestRegressor(n_estimators=10, min_samples_split=2, min_samples_leaf=1, max_features='log2', max_depth=5, bootstrap=True, random_state=0)
rf_B.fit(X_train_B, y_train_B)

# Random Forest for Location C
rf_C = RandomForestRegressor(n_estimators=10, min_samples_split=2, min_samples_leaf=1, max_features='log2', max_depth=5, bootstrap=True, random_state=0)
rf_C.fit(X_train_C, y_train_C)

# XGBoost for Location A
xgb_A = xgb.XGBRegressor(n_estimators=10, learning_rate=0.01, max_depth=10, random_state=0)
xgb_A.fit(X_train_A, y_train_A)

# XGBoost for Location B
xgb_B = xgb.XGBRegressor(n_estimators=10, learning_rate=0.01, max_depth=10, random_state=0)
xgb_B.fit(X_train_B, y_train_B)

# XGBoost for Location C
xgb_C = xgb.XGBRegressor(n_estimators=10, learning_rate=0.01, max_depth=10, random_state=0)
xgb_C.fit(X_train_C, y_train_C)

# Linear Regression for Location A
lr_A = LinearRegression()
lr_A.fit(X_train_A, y_train_A)

# Linear Regression for Location B
lr_B = LinearRegression()
lr_B.fit(X_train_B, y_train_B)

# Linear Regression for Location C
lr_C = LinearRegression()
lr_C.fit(X_train_C, y_train_C)

# StackingCVRegressor for Location A
stack_A = StackingCVRegressor(regressors=(rf_A, xgb_A, lr_A), meta_regressor=xgb_A, use_features_in_secondary=True)
stack_A.fit(X_train_A, y_train_A)

# StackingCVRegressor for Location B
stack_B = StackingCVRegressor(regressors=(rf_B, xgb_B, lr_B), meta_regressor=xgb_B, use_features_in_secondary=True)
stack_B.fit(X_train_B, y_train_B)

# StackingCVRegressor for Location C
stack_C = StackingCVRegressor(regressors=(rf_C, xgb_C, lr_C), meta_regressor=xgb_C, use_features_in_secondary=True)
stack_C.fit(X_train_C, y_train_C)

# Predict
pred_rf_A = rf_A.predict(X_test_A)
pred_rf_B = rf_B.predict(X_test_B)
pred_rf_C = rf_C.predict(X_test_C)

pred_xgb_A = xgb_A.predict(X_test_A)
pred_xgb_B = xgb_B.predict(X_test_B)
pred_xgb_C = xgb_C.predict(X_test_C)

pred_lr_A = lr_A.predict(X_test_A)
pred_lr_B = lr_B.predict(X_test_B)
pred_lr_C = lr_C.predict(X_test_C)

pred_stack_A = stack_A.predict(X_test_A)
pred_stack_B = stack_B.predict(X_test_B)
pred_stack_C = stack_C.predict(X_test_C)

# Evaluate
print('Random Forest')
print('Location A')
print('MAE:', mean_absolute_error(y_test_A, pred_rf_A))
print('MSE:', mean_squared_error(y_test_A, pred_rf_A))
print('R2:', r2_score(y_test_A, pred_rf_A))
print('Location B')
print('MAE:', mean_absolute_error(y_test_B, pred_rf_B))
print('MSE:', mean_squared_error(y_test_B, pred_rf_B))
print('R2:', r2_score(y_test_B, pred_rf_B))
print('Location C')
print('MAE:', mean_absolute_error(y_test_C, pred_rf_C))
print('MSE:', mean_squared_error(y_test_C, pred_rf_C))
print('R2:', r2_score(y_test_C, pred_rf_C))
print('XGBoost')
print('Location A')
print('MAE:', mean_absolute_error(y_test_A, pred_xgb_A))
print('MSE:', mean_squared_error(y_test_A, pred_xgb_A))
print('R2:', r2_score(y_test_A, pred_xgb_A))
print('Location B')
print('MAE:', mean_absolute_error(y_test_B, pred_xgb_B))
print('MSE:', mean_squared_error(y_test_B, pred_xgb_B))
print('R2:', r2_score(y_test_B, pred_xgb_B))
print('Location C')
print('MAE:', mean_absolute_error(y_test_C, pred_xgb_C))
print('MSE:', mean_squared_error(y_test_C, pred_xgb_C))
print('R2:', r2_score(y_test_C, pred_xgb_C))
print('Linear Regression')
print('Location A')
print('MAE:', mean_absolute_error(y_test_A, pred_lr_A))
print('MSE:', mean_squared_error(y_test_A, pred_lr_A))
print('R2:', r2_score(y_test_A, pred_lr_A))
print('Location B')
print('MAE:', mean_absolute_error(y_test_B, pred_lr_B))
print('MSE:', mean_squared_error(y_test_B, pred_lr_B))
print('R2:', r2_score(y_test_B, pred_lr_B))
print('Location C')
print('MAE:', mean_absolute_error(y_test_C, pred_lr_C))
print('MSE:', mean_squared_error(y_test_C, pred_lr_C))
print('R2:', r2_score(y_test_C, pred_lr_C))
print('StackingCVRegressor')
print('Location A')
print('MAE:', mean_absolute_error(y_test_A, pred_stack_A))
print('MSE:', mean_squared_error(y_test_A, pred_stack_A))
print('R2:', r2_score(y_test_A, pred_stack_A))
print('Location B')
print('MAE:', mean_absolute_error(y_test_B, pred_stack_B))
print('MSE:', mean_squared_error(y_test_B, pred_stack_B))
print('R2:', r2_score(y_test_B, pred_stack_B))
print('Location C')
print('MAE:', mean_absolute_error(y_test_C, pred_stack_C))
print('MSE:', mean_squared_error(y_test_C, pred_stack_C))
print('R2:', r2_score(y_test_C, pred_stack_C))

# Predict on actual values
X_A = pd.read_parquet('./Analysis/data/A/X_test_estimated.parquet')
X_B = pd.read_parquet('./Analysis/data/B/X_test_estimated.parquet')
X_C = pd.read_parquet('./Analysis/data/C/X_test_estimated.parquet')

# Keep columns
columns_to_keep = [
    "absolute_humidity_2m:gm3",
    "air_density_2m:kgm3",
    "clear_sky_energy_1h:J",
    "clear_sky_rad:W",
    "dew_point_2m:K",
    "diffuse_rad:W",
    "diffuse_rad_1h:J",
    "direct_rad:W",
    "direct_rad_1h:J",
    "effective_cloud_cover:p",
    "elevation:m",
    "is_day:idx",
    "is_in_shadow:idx",
    "msl_pressure:hPa",
    "pressure_100m:hPa",
    "pressure_50m:hPa",
    "relative_humidity_1000hPa:p",
    "sfc_pressure:hPa",
    "snow_water:kgm2",
    "sun_azimuth:d",
    "sun_elevation:d",
    "super_cooled_liquid_water:kgm2",
    "t_1000hPa:K",
    "total_cloud_cover:p",
    "visibility:m",
    "wind_speed_10m:ms",
    "wind_speed_u_10m:ms",
    "wind_speed_v_10m:ms"
]

X_A = X_A.set_index('date_forecast').resample('1H').mean()
X_B = X_B.set_index('date_forecast').resample('1H').mean()
X_C = X_C.set_index('date_forecast').resample('1H').mean()

X_A = keep_columns(X_A, columns_to_keep)
X_B = keep_columns(X_B, columns_to_keep)
X_C = keep_columns(X_C, columns_to_keep)

pred_cv_reg_A = stack_A.predict(X_A)
pred_cv_reg_B = stack_B.predict(X_B)
pred_cv_reg_C = stack_C.predict(X_C)

# Save predictions
pred_A = pd.DataFrame(pred_cv_reg_A)
pred_B = pd.DataFrame(pred_cv_reg_B)
pred_C = pd.DataFrame(pred_cv_reg_C)

# Concatenate to single prediction
pred = pd.concat([pred_A, pred_B, pred_C])

# Save to csv
pred.to_csv('./Analysis/data/pred.csv', index=True, header=True)