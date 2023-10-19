import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import pandas as pd

# Initialize H2O
h2o.init()

# Download logs
h2o.download_all_logs(dirname="Mathias/models/h2o_logs/")

# Load data
X_A = pd.read_parquet("preprocessing/data/est_A.parquet")

# Split X_A into training and validation sets
X_train_A, X_test_A = train_test_split(X_A, test_size=0.2, shuffle=False)

h2o_data_train = h2o.H2OFrame(X_train_A)
h2o_data_test = h2o.H2OFrame(X_test_A)

# Define target column
target = "pv_measurement"

# Define time column
time_col = "date_forecast"

# Define H2O AutoML model
aml = H2OAutoML(max_models=1, max_runtime_secs=14400, seed=1, max_runtime_secs_per_model=3600, exclude_algos=['XGBoost'])
# aml = H2OAutoML(seed=1)

# Train model
aml.train(x=h2o_data_train.columns, y=target, training_frame=h2o_data_train, leaderboard_frame=h2o_data_test, fold_column=time_col)


# View leaderboard
lb = aml.leaderboard
print(lb.head())