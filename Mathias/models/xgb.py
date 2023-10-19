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
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Define models
xgb_A = XGBRegressor(n_estimators=500, learning_rate=0.001, max_depth=20, random_state=3)
xgb_B = XGBRegressor(n_estimators=500, learning_rate=0.001, max_depth=20, random_state=3)
xgb_C = XGBRegressor(n_estimators=500, learning_rate=0.001, max_depth=20, random_state=3)
evals_results_A = {}
evals_results_B = {}
evals_results_C = {}

# Train the models
print('Training models...')
xgb_A.fit(X_A, y_A, eval_metric="mae", verbose=False)
print('A done')
xgb_B.fit(X_B, y_B, eval_metric="mae", verbose=False)
print('B done')
xgb_C.fit(X_C, y_C, eval_metric="mae", verbose=False)
print('C done')

# Fit the models
#xgb_A.fit(X_train_A, y_train_A, eval_set=[(X_train_A, y_train_A), (X_test_A, y_test_A)], eval_metric="mae", verbose=False)
#xgb_B.fit(X_train_B, y_train_B, eval_set=[(X_train_B, y_train_B), (X_test_B, y_test_B)], eval_metric="mae", verbose=False)
#xgb_C.fit(X_train_C, y_train_C, eval_set=[(X_train_C, y_train_C), (X_test_C, y_test_C)], eval_metric="mae", verbose=False)

# evals_results_A = xgb_A.evals_result()
# evals_results_B = xgb_B.evals_result()
# evals_results_C = xgb_C.evals_result()

# # Plotting training and validation errors
# train_errors = evals_results_A['validation_0']['mae']
# val_errors = evals_results_A['validation_1']['mae']
# plt.plot(train_errors, label='Train')
# plt.plot(val_errors, label='Validation')
# plt.xlabel('Boosting Round')
# plt.ylabel('MAE')
# plt.title('Training and Validation Errors for Model A')
# plt.legend()
# plt.show()

# Printing feature importances
# feature_importances = pd.DataFrame(xgb_A.feature_importances_,
#                                    index = X_train_A.columns,
#                                    columns=['importance']).sort_values('importance', ascending=False)
# print(feature_importances)

# Predict
pred_A = xgb_A.predict(test_A)
pred_B = xgb_B.predict(test_B)
pred_C = xgb_C.predict(test_C)

# Clip negative values to 0
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)

# Create submission
create_submission(pred_A, pred_B, pred_C, output_file="./Mathias/submission.csv")

