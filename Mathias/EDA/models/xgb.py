# Data libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

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
xgb_A = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=15, random_state=2)
xgb_B = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=15, random_state=2)
xgb_C = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=15, random_state=2)
evals_results_A = {}
evals_results_B = {}
evals_results_C = {}

# Train the models
# print('Training models...')
# xgb_A.fit(X_A, y_A, eval_metric="mae", verbose=False)
# print('A done')
# xgb_B.fit(X_B, y_B, eval_metric="mae", verbose=False)
# print('B done')
# xgb_C.fit(X_C, y_C, eval_metric="mae", verbose=False)
# print('C done')

# # Fit the models
print('Training models...')
xgb_A.fit(X_train_A, y_train_A, eval_set=[(X_train_A, y_train_A), (X_test_A, y_test_A)], eval_metric="mae", verbose=False)
xgb_A.save_model('./Mathias/EDA/saved_models/xgb_A.model')
print('A done')
xgb_B.fit(X_train_B, y_train_B, eval_set=[(X_train_B, y_train_B), (X_test_B, y_test_B)], eval_metric="mae", verbose=False)
xgb_B.save_model('./Mathias/EDA/saved_models/xgb_B.model')
print('B done')
xgb_C.fit(X_train_C, y_train_C, eval_set=[(X_train_C, y_train_C), (X_test_C, y_test_C)], eval_metric="mae", verbose=False)
xgb_C.save_model('./Mathias/EDA/saved_models/xgb_C.model')
print('C done')

# Predict
# pred_A = xgb_A.predict(test_A)
# pred_B = xgb_B.predict(test_B)
# pred_C = xgb_C.predict(test_C)

# Load models
xgb_A = XGBRegressor()
xgb_B = XGBRegressor()
xgb_C = XGBRegressor()

xgb_A.load_model('./Mathias/EDA/saved_models/xgb_A.model')
xgb_B.load_model('./Mathias/EDA/saved_models/xgb_B.model')
xgb_C.load_model('./Mathias/EDA/saved_models/xgb_C.model')

# Predict
y_pred_A = xgb_A.predict(X_test_A)
y_pred_B = xgb_B.predict(X_test_B)
y_pred_C = xgb_C.predict(X_test_C)

evals_results_A = xgb_A.evals_result()
evals_results_B = xgb_B.evals_result()
evals_results_C = xgb_C.evals_result()

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

# Evaluate the model
mae_A = mean_absolute_error(y_test_A, y_pred_A)
mae_B = mean_absolute_error(y_test_B, y_pred_B)
mae_C = mean_absolute_error(y_test_C, y_pred_C)

print(f"MAE A: {mae_A}")
print(f"MAE B: {mae_B}")
print(f"MAE C: {mae_C}")

# Plot predictions vs actual
y_pred_A = xgb_A.predict(X_test_A)
y_pred_A = np.array(y_pred_A)
y_test_A = np.array(y_test_A)
plt.plot(y_test_A, label='Actual')
plt.plot(y_pred_A, label='Predicted')
plt.xlabel('Time')
plt.ylabel('PV Power')
plt.title('Predictions vs Actual for Model A')
plt.legend()
plt.show()

y_pred_B = xgb_B.predict(X_test_B)
y_pred_B = np.array(y_pred_B)
y_test_B = np.array(y_test_B)
plt.plot(y_test_B, label='Actual')
plt.plot(y_pred_B, label='Predicted')
plt.xlabel('Time')
plt.ylabel('PV Power')
plt.title('Predictions vs Actual for Model B')
plt.legend()
plt.show()

y_pred_C = xgb_C.predict(X_test_C)
y_pred_C = np.array(y_pred_C)
y_test_C = np.array(y_test_C)
plt.plot(y_test_C, label='Actual')
plt.plot(y_pred_C, label='Predicted')
plt.xlabel('Time')
plt.ylabel('PV Power')
plt.title('Predictions vs Actual for Model C')
plt.legend()
plt.show()

# Printing feature importances
print('Feature importances:')
feature_importances_A = pd.DataFrame(xgb_A.feature_importances_,
                                   index = X_train_A.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances_A)

feature_importances_B = pd.DataFrame(xgb_B.feature_importances_,
                                   index = X_train_B.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances_B)

feature_importances_C = pd.DataFrame(xgb_C.feature_importances_,
                                      index = X_train_C.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances_C)

# Calculate residuals
residuals = y_test_A - y_pred_A

# Plotting residuals
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test_A, residuals, alpha=0.5)
plt.title('Residuals vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.title('Histogram of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Residual Statistics
print("Mean of Residuals: ", np.mean(residuals))
print("Standard Deviation of Residuals: ", np.std(residuals))
print("Root Mean Squared Error (RMSE): ", np.sqrt(mean_squared_error(y_test_A, y_pred_A)))

# Autocorrelation (Optional: requires statsmodels)
# It checks if residuals are correlated with their own lagged versions
try:
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals)
    plt.title('Autocorrelation of Residuals')
    plt.show()
except ImportError:
    print("statsmodels is not installed. Can't show autocorrelation plot.")

