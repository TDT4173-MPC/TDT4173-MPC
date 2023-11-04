# stack_fast.py

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from matplotlib import pyplot as plt

# Read in the data for A
data_path = './preprocessing/data'
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet')
test_A = pd.read_parquet(f'{data_path}/test_A.parquet').dropna()

# Concatenate A data
A = pd.concat([obs_A, est_A])

# Split to features and labels
X_A = A.drop(columns=['pv_measurement'])
y_A = A['pv_measurement']

# Sample data for quick training
sample_indices = X_A.sample(frac=0.1, random_state=42).index
sample_X_A = X_A.loc[sample_indices]
sample_y_A = y_A.loc[sample_indices]

X_train_A, X_val_A, y_train_A, y_val_A = train_test_split(sample_X_A, sample_y_A, test_size=0.2, random_state=42)

# Define base models with simplified hyperparameters and parallelization
base_models = [
    ('lr', LinearRegression(n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=150, criterion='absolute_error', n_jobs=-1)),
    ('ada', AdaBoostRegressor(n_estimators=300)),
    ('xgb', XGBRegressor(n_estimators=300, n_jobs=-1))
]

# Initialize an empty dictionary to hold trained base models for A
trained_base_models_A = {}

# Train base models and log progress
print('Training base models for A...')
for name, model in tqdm(base_models):
    model.fit(sample_X_A, sample_y_A)
    trained_base_models_A[name] = model
print('Base models for A are done.')

# Train stacking model for A
print('Training stack model for A...')
stack_A = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stack_A.fit(sample_X_A, sample_y_A)
print('Stack model for A is done.')

# Validation

y_pred = stack_A.predict(X_val_A)

# Calculate metrics
mae = mean_absolute_error(y_val_A, y_pred)

# Plot MAE
plt.figure(figsize=(6, 4))
plt.bar(['Validation'], [mae])
plt.xlabel('Dataset')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Model Performance')
plt.show()

print(f'Mean Absolute Error: {mae}')

