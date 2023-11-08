# Data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import pickle

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


class SARIMAXEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,24), model_path=''):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_path = model_path
        self.results = None

    def load_model(self):
        if self.model_path:
            self.results = pickle.load(open(self.model_path, 'rb'))

    def fit(self, X, y):
        if self.results is None:  # Only fit if the model hasn't been loaded
            self.model = SARIMAX(y, exog=X, order=self.order, seasonal_order=self.seasonal_order)
            self.results = self.model.fit()
        return self

    def predict(self, X):
        start = self.results.fittedvalues.shape[0]
        end = start + len(X) - 1
        return self.results.predict(start=start, end=end, exog=X)

columns =  ['direct_rad:W', 'effective_cloud_cover:p', 't_1000hPa:K',
            'total_cloud_cover:p', 'air_density_2m:kgm3', 'clear_sky_rad:W',
            'visibility:m', 'relative_humidity_1000hPa:p', 'wind_speed_u_10m:ms',
            'diffuse_rad_1h:J', 'wind_speed_v_10m:ms', 'sun_azimuth:d',
            'total_radiation', 'wind_vector_magnitude', 'average_wind_speed',
            'date_forecast_fft_amplitude', 'date_forecast_fft_phase',
            'sun_elevation:d_fft_phase', 'dew_point_2m:K_rate_of_change_of_change',
            't_1000hPa:K_rate_of_change', 'clear_sky_rad:W_rate_of_change',
            'clear_sky_rad:W_rate_of_change_of_change',
            'diffuse_rad:W_rate_of_change', 'direct_rad:W_rate_of_change',
            'effective_cloud_cover:p_rate_of_change_of_change']


# Read in the data
data_path = './preprocessing/data'
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet').drop(columns='date_forecast')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet').drop(columns='date_forecast')
obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet').drop(columns='date_forecast')
est_B = pd.read_parquet(f'{data_path}/est_B.parquet').drop(columns='date_forecast')
obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet').drop(columns='date_forecast')
est_C = pd.read_parquet(f'{data_path}/est_C.parquet').drop(columns='date_forecast')

test_A = pd.read_parquet(f'{data_path}/test_A.parquet').drop(columns='date_forecast').dropna()
test_B = pd.read_parquet(f'{data_path}/test_B.parquet').drop(columns='date_forecast').dropna()
test_C = pd.read_parquet(f'{data_path}/test_C.parquet').drop(columns='date_forecast').dropna()

# Concatenate
A = pd.concat([obs_A, est_A])
A['location'] = 1
B = pd.concat([obs_B, est_B])
B['location'] = 2
C = pd.concat([obs_C, est_C])
C['location'] = 3
X = pd.concat([A, B, C])
X = X.dropna()
y = X['pv_measurement']
X = X[columns]

test = pd.concat([test_A, test_B, test_C])
test = test[columns]

# A = A.replace([np.inf, -np.inf], np.nan)
# B = B.replace([np.inf, -np.inf], np.nan)
# C = C.replace([np.inf, -np.inf], np.nan)

# A = A.dropna()
# B = B.dropna()
# C = C.dropna()

# print(test_A.shape)
# print(test_B.shape)
# print(test_C.shape)

# Split to features and labels
# X_A = A.drop(columns=['pv_measurement'])
# y_A = A['pv_measurement']
# X_B = B.drop(columns=['pv_measurement'])
# y_B = B['pv_measurement']
# X_C = C.drop(columns=['pv_measurement'])
# y_C = C['pv_measurement']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
# X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
# X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Train models
# Define base models
lr = LinearRegression(n_jobs=-1)
rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, random_state=3)
xgb = XGBRegressor(n_estimators=150, learning_rate=0.01, max_depth=10, random_state=3)
dt = DecisionTreeRegressor(random_state=1, max_depth=10)
ada = AdaBoostRegressor(random_state=1, n_estimators=80, loss='linear', learning_rate=0.01)

# Initialize StackingRegressor with the base models and a meta-model
stack = StackingCVRegressor(regressors=(lr, rf, xgb, dt, ada), meta_regressor=LinearRegression(), cv=5, verbose=1)
stack_A = StackingCVRegressor(regressors=(lr, rf, xgb, dt, ada), meta_regressor=LinearRegression(), cv=5, verbose=1)
stack_B = StackingCVRegressor(regressors=(lr, rf, xgb, dt, ada), meta_regressor=LinearRegression(), cv=5, verbose=1)
stack_C = StackingCVRegressor(regressors=(lr, rf, xgb, dt, ada), meta_regressor=LinearRegression(), cv=5, verbose=1)

# Create a RandomForestRegressor
# estimator = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the RFE object and rank each feature
# selector = RFE(estimator, n_features_to_select=25, step=10, verbose=1)
# selector = selector.fit(X_train_A, y_train_A)

# Print the ranking of features
# print("Feature ranking:", selector.ranking_)

# Get the features that were selected by RFE
# selected_features = X_A.columns[selector.support_]
# print("Selected features:", selected_features)

# Train model
# stack.fit(X_train.values, y_train.values)
# pickle.dump(stack, open('./Mathias/EDA/saved_models/stack.pickle', 'wb'))
# print('MAE test: ', mean_absolute_error(y_test, stack.predict(X_test.values)))
# print('MAE train: ', mean_absolute_error(y_train, stack.predict(X_train.values)))


# Train the models
# print('Training models...')
# stack_A.fit(X_train_A.values, y_train_A.values)
# pickle.dump(stack_A, open('./Mathias/EDA/saved_models/stack_A.pickle', 'wb'))
# print('A done')
# stack_B.fit(X_train_B.values, y_train_B.values)
# pickle.dump(stack_B, open('./Mathias/EDA/saved_models/stack_B.pickle', 'wb'))
# print('B done')
# stack_C.fit(X_train_C.values, y_train_C.values)
# pickle.dump(stack_C, open('./Mathias/EDA/saved_models/stack_C.pickle', 'wb'))
# print('C done')

# Load models
stack = pickle.load(open('./Mathias/EDA/saved_models/stack.pickle', 'rb'))
# stack_A = pickle.load(open('./Mathias/EDA/saved_models/stack_A.pickle', 'rb'))
# stack_B = pickle.load(open('./Mathias/EDA/saved_models/stack_B.pickle', 'rb'))
# stack_C = pickle.load(open('./Mathias/EDA/saved_models/stack_C.pickle', 'rb'))

# Predict
print('Predicting...')
pred = stack.predict(X_test.values)
# pred_A = stack_A.predict(X_train_A)
# pred_B = stack_B.predict(X_test_B)
# pred_C = stack_C.predict(X_test_C)

# Clip negative values to 0
pred = np.clip(pred, 0, None)
# pred_A = np.clip(pred_A, 0, None)
# pred_B = np.clip(pred_B, 0, None)
# pred_C = np.clip(pred_C, 0, None)

# Plotting predicted value vs true value
plt.figure(figsize=(10,6))
plt.plot(pred, label='pred')
plt.plot(y_test.values, label='actual')
plt.legend()
plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(pred_A, label='pred')
# plt.plot(y_train_A.values, label='actual')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(pred_B, label='pred')
# plt.plot(y_test_B.values, label='actual')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,6))
# plt.plot(pred_C, label='pred')
# plt.plot(y_test_C.values, label='actual')
# plt.legend()
# plt.show()

# Evaluate
# print('Evaluating...')
# mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
# cv_scores_A = -cross_val_score(stack_A, X_train_A.values, y_train_A.values, cv=5, scoring=mae_scorer, verbose=1, n_jobs=-1)
# mean_cv_mae_A = np.mean(cv_scores_A)
# print('Average MAE with CV - A:', mean_cv_mae_A)

# pred_A = stack_A.predict(X_train_A.values)
# pred_B = stack_B.predict(X_train_B.values)
# pred_C = stack_C.predict(X_train_C.values)
# print('MAE train A:', mean_absolute_error(y_train_A, pred_A))
# print('MAE train B:', mean_absolute_error(y_train_B, pred_B))
# print('MAE train C:', mean_absolute_error(y_train_C, pred_C))
# pred_A = stack_A.predict(X_test_A.values)
# pred_B = stack_B.predict(X_test_B.values)
# pred_C = stack_C.predict(X_test_C.values)
# print('MAE test A:', mean_absolute_error(y_test_A, pred_A))
# print('MAE test B:', mean_absolute_error(y_test_B, pred_B))
# print('MAE test C:', mean_absolute_error(y_test_C, pred_C))

# Write pred_A to file
# pred_A = stack_A.predict(test_A.values)
# pred_A = np.clip(pred_A, 0, None)
# pred_A = pd.DataFrame(pred_A)
# pred_A.to_csv('./Mathias/EDA/pred_A.csv', index=False)

# pred_B = stack_B.predict(test_B.values)
# pred_B = np.clip(pred_B, 0, None)
# pred_B = pd.DataFrame(pred_B)
# pred_B.to_csv('./Mathias/EDA/pred_B.csv', index=False)

# pred_C = stack_C.predict(test_C.values)
# pred_C = np.clip(pred_C, 0, None)
# pred_C = pd.DataFrame(pred_C)
# pred_C.to_csv('./Mathias/EDA/pred_C.csv', index=False)

pred = stack.predict(test.values)
pred = np.clip(pred, 0, None)

ids = np.arange(0, len(pred))

# Create a DataFrame
df = pd.DataFrame({
    'id': ids,
    'prediction': pred
})

output_file = "submission_lol.csv"

# Save to CSV
df.to_csv(output_file, index=False)
print(f"Submission saved to {output_file}")


# # Load predictions
# pred_A = pd.read_csv('./Mathias/EDA/pred_A.csv')
# pred_A = pred_A['0']
# pred_B = pd.read_csv('./Mathias/EDA/pred_B.csv')
# pred_B = pred_B['0']
# pred_C = pd.read_csv('./Mathias/EDA/pred_C.csv')
# pred_C = pred_C['0']

# # Create submission
# create_submission(pred_A, pred_B, pred_C, output_file="submission.csv")

