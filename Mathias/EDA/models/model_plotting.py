# Data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import mean_absolute_error
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

# Import models
stack_A = pickle.load(open('Mathias/EDA/saved_models/stack_A.model', 'rb'))
stack_B = pickle.load(open('Mathias/EDA/saved_models/stack_B.model', 'rb'))
stack_C = pickle.load(open('Mathias/EDA/saved_models/stack_C.model', 'rb'))

# Predict
pred_A = stack_A.predict(X_test_A)
pred_B = stack_B.predict(X_test_B)
pred_C = stack_C.predict(X_test_C)

# Clip negative values to 0
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)

# Calculate MAE
print('MAE A:', mean_absolute_error(y_test_A, pred_A))
print('MAE B:', mean_absolute_error(y_test_B, pred_B))
print('MAE C:', mean_absolute_error(y_test_C, pred_C))

# Plot predictions vs. actual values
# Plot the predictions
# plt.plot(pred_A, label='pred')
# plt.plot(y_test_A.values, label='actual')
# plt.legend()
# plt.show()

# Create submission
pred_A = stack_A.predict(test_A)
pred_B = stack_B.predict(test_B)
pred_C = stack_C.predict(test_C)
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)
create_submission(pred_A, pred_B, pred_C, output_file="submission.csv")


