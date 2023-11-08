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


# Print model coefficients and intercept
def print_coefficients(model, feature_names, model_name):
    print(f"Model {model_name} intercept: {model.intercept_}")
    coefficients = dict(zip(feature_names, model.coef_))
    for feature in coefficients:
        print(f"Coefficient for {feature}: {coefficients[feature]}")


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

# Split into training and validation sets
train_A, val_A = train_test_split(A, test_size=0.2, shuffle=False)
train_B, val_B = train_test_split(B, test_size=0.2, shuffle=False)
train_C, val_C = train_test_split(C, test_size=0.2, shuffle=False)

# Split to features and labels
X_train_A = train_A.drop(columns=['pv_measurement'])
y_train_A = train_A['pv_measurement']
X_val_A = val_A.drop(columns=['pv_measurement'])
y_val_A = val_A['pv_measurement']

X_train_B = train_B.drop(columns=['pv_measurement'])
y_train_B = train_B['pv_measurement']
X_val_B = val_B.drop(columns=['pv_measurement'])
y_val_B = val_B['pv_measurement']

X_train_C = train_C.drop(columns=['pv_measurement'])
y_train_C = train_C['pv_measurement']
X_val_C = val_C.drop(columns=['pv_measurement'])
y_val_C = val_C['pv_measurement']

# Train the model
model_A = LinearRegression()
model_B = LinearRegression()
model_C = LinearRegression()

model_A.fit(X_train_A, y_train_A)
model_B.fit(X_train_B, y_train_B)
model_C.fit(X_train_C, y_train_C)

# Make predictions
pred_A = model_A.predict(X_val_A)
pred_B = model_B.predict(X_val_B)
pred_C = model_C.predict(X_val_C)

# Cap negative prediction to zero
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)

# Evaluate the model
mae_A = mean_absolute_error(y_val_A, pred_A)
mae_B = mean_absolute_error(y_val_B, pred_B)
mae_C = mean_absolute_error(y_val_C, pred_C)

print(f"MAE A: {mae_A}")
print(f"MAE B: {mae_B}")
print(f"MAE C: {mae_C}")

# Assuming that your features are the same across A, B, and C, we can use the column names from X_train_A
feature_names = X_train_A.columns.tolist()

# print("=== Model A ===")
# print_coefficients(model_A, feature_names, "A")

# print("\n=== Model B ===")
# print_coefficients(model_B, feature_names, "B")

# print("\n=== Model C ===")
# print_coefficients(model_C, feature_names, "C")

# Plot the predictions
# plt.plot(pred_A, label='pred')
# plt.plot(y_val_A.values, label='actual')
# plt.legend()
# plt.show()

# Score with 'is_in_shadow:idx'
# MAE A: 274.07583403745235
# MAE B: 33.87757171808925
# MAE C: 29.35363008180999

# Score without 'is_in_shadow:idx'
# MAE A: 274.07583403745235
# MAE B: 33.87757171808925
# MAE C: 29.35363008180999