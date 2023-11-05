# stack_fast.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from matplotlib import pyplot as plt

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


def read_and_preprocess(data_path, data_name, sample_frac=1.0):
    obs = pd.read_parquet(f'{data_path}/obs_{data_name}.parquet')
    est = pd.read_parquet(f'{data_path}/est_{data_name}.parquet')
    test = pd.read_parquet(f'{data_path}/test_{data_name}.parquet').dropna()
    combined = pd.concat([obs, est])
    
    if sample_frac < 1.0:
        sample_indices = combined.sample(frac=sample_frac, random_state=42).index
        combined = combined.loc[sample_indices]
        
    X = combined.drop(columns=['pv_measurement'])
    y = combined['pv_measurement']
    return X, y, test

# Read and preprocess data
data_path = './preprocessing/data'
X_A, y_A, test_A = read_and_preprocess(data_path, 'A', sample_frac=0.1)
X_B, y_B, test_B = read_and_preprocess(data_path, 'B', sample_frac=0.1)
X_C, y_C, test_C = read_and_preprocess(data_path, 'C', sample_frac=0.1)


# Define base models
base_models = [
    ('lr', LinearRegression(n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=200, criterion='absolute_error', n_jobs=-1)),
    ('ada', AdaBoostRegressor(n_estimators=300)),
    ('xgb', XGBRegressor(n_estimators=300, n_jobs=-1))
]

# Initialize StackingRegressor
stack_A = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stack_B = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
stack_C = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Training
print('Training models for A, B, and C...')
with tqdm(total=3, desc="Training Stacking Models") as pbar:
    stack_A.fit(X_A, y_A)
    pbar.update(1)
    print('A done')
    
    stack_B.fit(X_B, y_B)
    pbar.update(1)
    print('B done')
    
    stack_C.fit(X_C, y_C)
    pbar.update(1)
    print('C done')

# Generate predictions
pred_A = stack_A.predict(test_A)
pred_B = stack_B.predict(test_B)
pred_C = stack_C.predict(test_C)

# Clip negative values
pred_A = np.clip(pred_A, 0, None)
pred_B = np.clip(pred_B, 0, None)
pred_C = np.clip(pred_C, 0, None)

# Create submission (you should define this function in your code)
create_submission(pred_A, pred_B, pred_C, output_file="./Mathias/submission.csv")

# Validation (for A as an example)
y_pred_A = stack_A.predict(X_A)
mae = mean_absolute_error(y_A, y_pred_A)

# Plotting
plt.figure(figsize=(6, 4))
plt.bar(['Validation'], [mae])
plt.xlabel('Dataset')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Model Performance')
plt.show()

print(f'Mean Absolute Error: {mae}')
