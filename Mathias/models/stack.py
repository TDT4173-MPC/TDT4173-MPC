# Data libraries
import pandas as pd
import numpy as np

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

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

def keep_columns(df, columns):
    df = df[columns]
    return df

# Define columns to keep
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


# Read in the data
data_path = 'Analysis/preprocessing/data'
obs_A = pd.read_csv(f'{data_path}/obs_A.csv')
est_A = pd.read_csv(f'{data_path}/est_A.csv')
obs_B = pd.read_csv(f'{data_path}/obs_B.csv')
est_B = pd.read_csv(f'{data_path}/est_B.csv')
obs_C = pd.read_csv(f'{data_path}/obs_C.csv')
est_C = pd.read_csv(f'{data_path}/est_C.csv')

test_A = pd.read_parquet('Analysis/data/A/X_test_estimated.parquet')
test_B = pd.read_parquet('Analysis/data/B/X_test_estimated.parquet')
test_C = pd.read_parquet('Analysis/data/C/X_test_estimated.parquet')

test_A = test_A.set_index('date_forecast').resample('1H').mean()
test_B = test_B.set_index('date_forecast').resample('1H').mean()
test_C = test_C.set_index('date_forecast').resample('1H').mean()

test_A = keep_columns(test_A, columns_to_keep).dropna()
test_B = keep_columns(test_B, columns_to_keep).dropna()
test_C = keep_columns(test_C, columns_to_keep).dropna()

# Concatenate
A = pd.concat([obs_A, est_A])
B = pd.concat([obs_B, est_B])
C = pd.concat([obs_C, est_C])

# Split to features and labels
X_A = A.drop(columns=['pv_measurement'])
y_A = A['pv_measurement']
X_B = B.drop(columns=['pv_measurement'])
y_B = B['pv_measurement']
X_C = C.drop(columns=['pv_measurement'])
y_C = C['pv_measurement']

# Split into train and test
# X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
# X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
# X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Train models
# Define base models
base_models = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor()),
    ('ada', AdaBoostRegressor()),
    ('xgb', XGBRegressor())
]

# Initialize StackingRegressor with the base models and a meta-model
stack_A = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stack_B = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stack_C = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Train the models
# print('Training models...')
# stack_A.fit(X_A, y_A)
# print('A done')
# stack_B.fit(X_B, y_B)
# print('B done')
# stack_C.fit(X_C, y_C)
# print('C done')

print(test_A.shape)
print(test_B.shape)
print(test_C.shape)

# Predict
# pred_A = stack_A.predict(test_A)
# pred_B = stack_B.predict(test_B)
# pred_C = stack_C.predict(test_C)


# Evaluate


# Create submission
#create_submission(pred_A, pred_B, pred_C, output_file="Analysis/Mathias/submission.csv")

# # Read the CSV file
# df = pd.read_csv('Analysis/Mathias/submission.csv')

# # Check if 'id' column exists in the dataframe
# if 'id' in df.columns:
#     df['id'] = df['id'] - 1

# # Save the modified dataframe back to the same CSV file
# df.to_csv('Analysis/Mathias/submission.csv', index=False)
