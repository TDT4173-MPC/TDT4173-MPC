# Data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Models
from xgboost import XGBRegressor
from xgboost import plot_importance

# Utils
def create_submission(pred_A, pred_B, pred_C, output_file="submission.csv"):
    """
    Create a Kaggle submission file.

    Parameters:
    - pred_A, pred_B, pred_C: Arrays containing predictions.
    - output_file: Name of the output CSV file.

    Returns:
    - None. Writes the submission to a CSV file.
    """
    
    # Concatenate predictions
    predictions = np.concatenate([pred_A, pred_B, pred_C])

    # Create an id array
    ids = np.arange(0, len(predictions))

    # Create a DataFrame
    df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


# Read in the data
data_path = './preprocessing/data'


columns_A = [
    "pv_measurement",
    "direct_rad:W",
    "diffuse_rad:W",
    "direct_rad:W_rate_of_change",
    "clear_sky_rad:W",
    #"date_forecast_fft_amplitude",
    "clear_sky_rad:W_rate_of_change_of_change",
    "clear_sky_rad:W_rate_of_change",
    "sun_azimuth:d",
    "direct_rad:W_rate_of_change_of_change",
    "diffuse_rad_1h:J",
    "t_1000hPa:K",
    "precip_5min:mm",
    "msl_pressure:hPa",
    "sun_elevation:d",
    #"sun_elevation:d_fft_phase",
    "t_1000hPa:K_rate_of_change",
    "fresh_snow_24h:cm",
    "diffuse_rad:W_rate_of_change",
    "direct_rad_1h:J",
    "absolute_humidity_2m:gm3",
    "precipitation"
]
columns_B = [
    "pv_measurement",
    #"date_forecast_fft_phase",
    "direct_rad:W",
    "diffuse_rad:W",
    "sun_elevation:d",
    "clear_sky_rad:W",
    "clear_sky_rad:W_rate_of_change",
    #"date_forecast_fft_amplitude",
    #"cloud_base_agl:m",
    #"year",
    "t_1000hPa:K",
    #"snow_drift:idx_fft_amplitude",
    "air_density_2m:kgm3",
    "diffuse_rad:W_rate_of_change",
    "clear_sky_rad:W_rate_of_change_of_change",
    "t_1000hPa:K_rate_of_change",
    #"month",
    "diffuse_rad_1h:J",
    "direct_rad:W_rate_of_change",
    "visibility:m",
    "precip_5min:mm",
    "precipitation"
]
columns_C = [
    "pv_measurement",
    "direct_rad:W",
    "sun_elevation:d",
    "diffuse_rad:W",
    "t_1000hPa:K",
    "direct_rad_1h:J",
    #"date_forecast_fft_amplitude",
    "clear_sky_rad:W",
    "clear_sky_energy_1h:J",
    "direct_rad:W_rate_of_change_of_change",
    "snow_melt_10min:mm",
    "direct_rad:W_rate_of_change",
    "precip_5min:mm",
    "relative_humidity_1000hPa:p",
    "msl_pressure:hPa",
    #"precip_type_5min:idx_fft_amplitude",
    "wind_speed_u_10m:ms",
    "diffuse_rad_1h:J",
    "sfc_pressure:hPa",
    "dew_point_2m:K",
    "effective_cloud_cover:p",
    "precipitation"
]

# For A
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet')
A = pd.concat([obs_A, est_A])

# For B
obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet')
est_B = pd.read_parquet(f'{data_path}/est_B.parquet')
B = pd.concat([obs_B, est_B])


# For C
obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet')
est_C = pd.read_parquet(f'{data_path}/est_C.parquet')
C = pd.concat([obs_C, est_C])


# For testing, read in test data and select only the specified columns, then drop missing values
test_A = pd.read_parquet(f'{data_path}/test_A.parquet')
test_B = pd.read_parquet(f'{data_path}/test_B.parquet')
test_C = pd.read_parquet(f'{data_path}/test_C.parquet')

A = A.loc[:, columns_A]

# Select only the columns specified in columns_B for dataset B
B = B.loc[:, columns_B]

# Select only the columns specified in columns_C for dataset C
C = C.loc[:, columns_C]

# Also, apply the same column filtering for your test sets:
features_A = [col for col in columns_A if col != "pv_measurement"]
features_B = [col for col in columns_B if col != "pv_measurement"]
features_C = [col for col in columns_C if col != "pv_measurement"]

# Now use these feature lists to select columns from the test data
test_A = test_A[features_A]
test_B = test_B[features_B]
test_C = test_C[features_C]

print(test_A.shape)

# Split to features and labels
X_A = A.drop(columns=['pv_measurement'])
y_A = A['pv_measurement']
X_B = B.drop(columns=['pv_measurement'])
y_B = B['pv_measurement']
X_C = C.drop(columns=['pv_measurement'])
y_C = C['pv_measurement']

# Split into train and test
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Define models
xgb_A = XGBRegressor(n_estimators=600, learning_rate=0.009, max_depth=10, random_state=0)
xgb_B = XGBRegressor(n_estimators=600, learning_rate=0.009, max_depth=10, random_state=0)
xgb_C = XGBRegressor(n_estimators=600, learning_rate=0.009, max_depth=10, random_state=0)
evals_results_A = {}
evals_results_B = {}
evals_results_C = {}

# Train the models
print('Training models...')
xgb_A.fit(X_A, y_A)
print('A done')
xgb_B.fit(X_B, y_B)
print('B done')
xgb_C.fit(X_C, y_C)
print('C done')

# Fit the models
xgb_A.fit(X_train_A, y_train_A, eval_set=[(X_train_A, y_train_A), (X_test_A, y_test_A)], eval_metric="mae", verbose=True)
xgb_B.fit(X_train_B, y_train_B, eval_set=[(X_train_B, y_train_B), (X_test_B, y_test_B)], eval_metric="mae", verbose=True)
xgb_C.fit(X_train_C, y_train_C, eval_set=[(X_train_C, y_train_C), (X_test_C, y_test_C)], eval_metric="mae", verbose=True)

evals_results_A = xgb_A.evals_result()
evals_results_B = xgb_B.evals_result()
evals_results_C = xgb_C.evals_result()

# Get the importance dictionary (by default, 'weight' is used)
importance_dict = xgb_A.get_score(importance_type='weight')

# Plotting training and validation errors
train_errors = evals_results_A['validation_0']['mae']
val_errors = evals_results_A['validation_1']['mae']
plt.plot(train_errors, label='Train')
plt.plot(val_errors, label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('MAE')
plt.title('Training and Validation Errors for Model A')
plt.legend()
plt.show()

# Predict
pred_A = xgb_A.predict(test_A)
pred_B = xgb_B.predict(test_B)
pred_C = xgb_C.predict(test_C)

# Clip negative values to 0
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)

# Get predictions for the test set (already have y_test_A for actual values)
y_pred_A_test = xgb_A.predict(X_test_A)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_A, y_pred_A_test, alpha=0.7)
plt.xlabel('Actual A values')
plt.ylabel('Predicted A values')
plt.title('Actual vs Predicted A values')
plt.plot([min(y_test_A), max(y_test_A)], [min(y_test_A), max(y_test_A)], color='red')  # Diagonal line
plt.show()

# Create submission
create_submission(pred_A, pred_B, pred_C, output_file="./analysis/submission.csv")

