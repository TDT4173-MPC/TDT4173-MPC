# Data libraries
import pandas as pd
import numpy as np
from tqdm import tqdm  # for the progress bar

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
A = pd.concat([obs_A, est_A])
# B = pd.concat([obs_B, est_B])
# C = pd.concat([obs_C, est_C])

# Split to features and labels
X_A = A.drop(columns=['pv_measurement'])
y_A = A['pv_measurement']
# X_B = B.drop(columns=['pv_measurement'])
# y_B = B['pv_measurement']
# X_C = C.drop(columns=['pv_measurement'])
# y_C = C['pv_measurement']

# Split into train and test
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, shuffle=False)
# X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, shuffle=False)
# X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(X_C, y_C, test_size=0.2, shuffle=False)

# Train models
# Define base models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define base models
base_models = [
    ('lr', LinearRegression(n_jobs=-1)),
    ('rf', RandomForestRegressor(n_jobs=-1)),
    ('ada', AdaBoostRegressor()),
    ('xgb', XGBRegressor(n_jobs=-1))
]

# Define hyperparameters for tuning (simplified for illustration)
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__criterion': ['mse', 'mae'],
    'ada__n_estimators': [50, 100, 150],
    'xgb__n_estimators': [100, 200, 300],
    'final_estimator__n_estimators': [50, 100, 150]
}

# Initialize StackingRegressor with the base models and a meta-model
final_estimator = RandomForestRegressor(n_estimators=100)

# Create StackingRegressor instances
stack_A = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
# stack_B = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
# stack_C = StackingRegressor(estimators=base_models, final_estimator=final_estimator)

# Create GridSearchCV instances for hyperparameter tuning
grid_A = GridSearchCV(estimator=stack_A, param_grid=param_grid, cv=5)
# grid_B = GridSearchCV(estimator=stack_B, param_grid=param_grid, cv=5)
# grid_C = GridSearchCV(estimator=stack_C, param_grid=param_grid, cv=5)

# Fit the models (simplified for illustration)
grid_A.fit(X_A, y_A)
# grid_B.fit(X_B, y_B)
# grid_C.fit(X_C, y_C)

# Optionally, extract the best estimators after hyperparameter tuning
best_stack_A = grid_A.best_estimator_
# best_stack_B = grid_B.best_estimator_
# best_stack_C = grid_C.best_estimator_


# Predict
pred_A = best_stack_A.predict(test_A)
# pred_B = best_stack_B.predict(test_B)
# pred_C = best_stack_C.predict(test_C)

# Clip negative values to 0
pred_A = np.clip(pred_A, 0, None)
# pred_B = np.clip(pred_B, 0, None)
# pred_C = np.clip(pred_C, 0, None)

# Create submission
# create_submission(pred_A, output_file="Analysis/Mathias/submission.csv")

