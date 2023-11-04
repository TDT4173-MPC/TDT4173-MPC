
# Data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Models
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet')
obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet')
est_B = pd.read_parquet(f'{data_path}/est_B.parquet')
obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet')
est_C = pd.read_parquet(f'{data_path}/est_C.parquet')

test_A = pd.read_parquet(f'{data_path}/test_A.parquet').dropna()
test_B = pd.read_parquet(f'{data_path}/test_B.parquet').dropna()
test_C = pd.read_parquet(f'{data_path}/test_C.parquet').dropna()

# Concatenate
# A = pd.concat([obs_A, est_A])
# B = pd.concat([obs_B, est_B])
# C = pd.concat([obs_C, est_C])
A = obs_A
B = obs_B
C = obs_C

# Split to features and labels
X_A = A.drop(columns=['pv_measurement'])
y_A = A['pv_measurement']
X_B = B.drop(columns=['pv_measurement'])
y_B = B['pv_measurement']
X_C = C.drop(columns=['pv_measurement'])
y_C = C['pv_measurement']

# Split into train and test
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
# X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
# X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Initialize SARIMAX models
sarimax_A = SARIMAX(y_A, exog=X_A, order=(1,1,1), seasonal_order=(1,1,1,24))
# sarimax_B = SARIMAX(y_B, exog=X_B, order=(1,1,1), seasonal_order=(1,1,1,24))
# sarimax_C = SARIMAX(y_C, exog=X_C, order=(1,1,1), seasonal_order=(1,1,1,24))

# Save memory
# del A, B, C

# Fit the models
# print('Fitting models...')

# sarimax_A = sarimax_A.fit()
# pickle.dump(sarimax_A, open('./Mathias/models/weights/sarimax_A.sav', 'wb'))
# del sarimax_A
# print('A done')

# sarimax_B = sarimax_B.fit()
# pickle.dump(sarimax_B, open('./Mathias/models/weights/sarimax_B.sav', 'wb'))
# del sarimax_B
# print('B done')

# sarimax_C = sarimax_C.fit()
# pickle.dump(sarimax_C, open('./Mathias/models/weights/sarimax_C.sav', 'wb'))
# del sarimax_C
# print('C done')

# Load the models
# print('Loading models...')
sarimax_A = pickle.load(open('./Mathias/models/weights/sarimax_A.sav', 'rb'))
sarimax_B = pickle.load(open('./Mathias/models/weights/sarimax_B.sav', 'rb'))
sarimax_C = pickle.load(open('./Mathias/models/weights/sarimax_C.sav', 'rb'))

# Predict
# Predict
start_A = len(y_A)
end_A = start_A + len(test_A) - 1
forecast_A = sarimax_A.predict(start=start_A, end=end_A, exog=test_A)

start_B = len(y_B)
end_B = start_B + len(test_B) - 1
forecast_B = sarimax_B.predict(start=start_B, end=end_B, exog=test_B)

start_C = len(y_C)
end_C = start_C + len(test_C) - 1
forecast_C = sarimax_C.predict(start=start_C, end=end_C, exog=test_C)


# sarimax_A.plot_diagnostics(figsize=(15, 12))
# plt.show()
# sarimax_B.plot_diagnostics(figsize=(15, 12))
# plt.show()
# sarimax_C.plot_diagnostics(figsize=(15, 12))
# plt.show()

# Plot the predictions
# plt.plot(forecast_A, label='A')
# plt.plot(forecast_B, label='B')
# plt.plot(forecast_C, label='C')
# plt.legend()
# plt.show()

# Clip negative values to 0
forecast_A = np.clip(forecast_A, 0, None)
forecast_B = np.clip(forecast_B, 0, None)
forecast_C = np.clip(forecast_C, 0, None)

# Create submission
create_submission(forecast_A, forecast_B, forecast_C, output_file="./Mathias/submission.csv")

print('Done')
