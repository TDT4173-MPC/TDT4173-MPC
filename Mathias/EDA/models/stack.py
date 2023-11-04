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



# Read in the data
data_path = './preprocessing/data'
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet').drop(columns='date_forecast')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet').drop(columns='date_forecast')
obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet').drop(columns='date_forecast')
est_B = pd.read_parquet(f'{data_path}/est_B.parquet').drop(columns='date_forecast')
obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet').drop(columns='date_forecast')
est_C = pd.read_parquet(f'{data_path}/est_C.parquet').drop(columns='date_forecast')

test_A = pd.read_parquet(f'{data_path}/test_A.parquet').dropna().drop(columns='date_forecast')
test_B = pd.read_parquet(f'{data_path}/test_B.parquet').dropna().drop(columns='date_forecast')
test_C = pd.read_parquet(f'{data_path}/test_C.parquet').dropna().drop(columns='date_forecast')

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
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Train models
# Define base models
lr = LinearRegression(n_jobs=-1)
rf = RandomForestRegressor(n_jobs=-1, n_estimators=120, random_state=1)
xgb = XGBRegressor(n_estimators=250, learning_rate=0.01, max_depth=15, random_state=1)
dt = DecisionTreeRegressor(random_state=1, max_depth=15)
ada = AdaBoostRegressor(random_state=1, n_estimators=100)

# Initialize StackingRegressor with the base models and a meta-model
stack_A = StackingCVRegressor(regressors=(lr, rf, xgb, dt, ada), meta_regressor=LinearRegression(), cv=5, verbose=1)
stack_B = StackingCVRegressor(regressors=(lr, rf, xgb), meta_regressor=LinearRegression(), cv=5)
stack_C = StackingCVRegressor(regressors=(lr, rf, xgb), meta_regressor=LinearRegression(), cv=5)

# Train the models
# print('Training models...')
# stack_A.fit(X_A.values, y_A.values)
# pickle.dump(stack_A, open('./Mathias/EDA/saved_models/stack_A.pickle', 'wb'))
# print('A done')
# stack_B.fit(X_train_B.values, y_train_B.values)
# pickle.dump(stack_B, open('./Mathias/EDA/saved_models/stack_B.pickle', 'wb'))
# print('B done')
# stack_C.fit(X_train_C.values, y_train_C.values)
# pickle.dump(stack_C, open('./Mathias/EDA/saved_models/stack_C.pickle', 'wb'))
# print('C done')

# Load models
stack_A = pickle.load(open('./Mathias/EDA/saved_models/stack_A.pickle', 'rb'))
# stack_B = pickle.load(open('./Mathias/EDA/saved_models/stack_B.pickle', 'rb'))
# stack_C = pickle.load(open('./Mathias/EDA/saved_models/stack_C.pickle', 'rb'))

# Predict
pred_A = stack_A.predict(X_test_A)
# pred_B = stack_B.predict(X_test_B)
# pred_C = stack_C.predict(X_test_C)

# Clip negative values to 0
pred_A = np.clip(pred_A, 0, None)
# pred_B = np.clip(pred_B, 0, None)
# pred_C = np.clip(pred_C, 0, None)

# Plotting predicted value vs true value
# plt.figure(figsize=(10,6))
# plt.plot(pred_A, label='pred')
# plt.plot(y_test_A.values, label='actual')
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
print('Evaluating...')
print('MAE A:', mean_absolute_error(y_test_A, pred_A))
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
cv_scores_A = -cross_val_score(stack_A, X_train_A.values, y_train_A.values, cv=5, scoring=mae_scorer, verbose=1, n_jobs=-1)
mean_cv_mae_A = np.mean(cv_scores_A)
print('Average MAE with CV - A:', mean_cv_mae_A)
# print('B:', mean_absolute_error(y_test_B, pred_B))
# print('C:', mean_absolute_error(y_test_C, pred_C))

# Write pred_A to file
pred_A = stack_A.predict(test_A.values)
pred_A = np.clip(pred_A, 0, None)
pred_A = pd.DataFrame(pred_A)
pred_A.to_csv('./Mathias/EDA/pred_A.csv', index=False)

# pred_B = stack_B.predict(test_B.values)
# pred_B = np.clip(pred_B, 0, None)
# pred_B = pd.DataFrame(pred_B)
# pred_B.to_csv('./Mathias/EDA/pred_B.csv', index=False)

# pred_C = stack_C.predict(test_C.values)
# pred_C = np.clip(pred_C, 0, None)
# pred_C = pd.DataFrame(pred_C)
# pred_C.to_csv('./Mathias/EDA/pred_C.csv', index=False)

# Load predictions
pred_A = pd.read_csv('./Mathias/EDA/pred_A.csv')
pred_A = pred_A['0']
pred_B = pd.read_csv('./Mathias/EDA/pred_B.csv')
pred_B = pred_B['0']
pred_C = pd.read_csv('./Mathias/EDA/pred_C.csv')
pred_C = pred_C['0']

# Create submission
create_submission(pred_A, pred_B, pred_C, output_file="submission.csv")

