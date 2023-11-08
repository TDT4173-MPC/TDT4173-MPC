import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import d2_absolute_error_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import PredictionErrorDisplay

def test_xgboost(params, train, test):

    # Prepare data
    X_train = train.drop(columns='pv_measurement')
    X_test  = test.drop(columns='pv_measurement')
    y_train = train['pv_measurement']
    y_test  = test['pv_measurement']

    # Fit model
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(X=X_train, y=y_train)

    # Evaluate 
    pred = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    medae = median_absolute_error(y_test, pred)
    max_err = max_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    d2 = d2_absolute_error_score(y_test, pred)

    # Get parameters
    params = xgb_model.get_params()

    # Get feature importances and pair them with feature names
    feature_importances = xgb_model.feature_importances_
    feature_names = X_train.columns.tolist()
    features_and_importances = {fname: float(fimportance) for fname, fimportance in zip(feature_names, feature_importances)}

    # Sort the features by their importance in descending order
    sorted_features_and_importances = dict(sorted(features_and_importances.items(), key=lambda item: item[1], reverse=True))

    # Ensure all parameters are serializable
    serializable_params = {key: str(value) for key, value in params.items()}

    results = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'medae': float(medae),
        'max_error': float(max_err),
        'r2_score': float(r2),
        'd2_score': float(d2),
        'feature_importances': sorted_features_and_importances,
        'params': serializable_params
    }

    # Visualize Prediction Error
    # from sklearn.metrics import PredictionErrorDisplay
    # fig, ax = plt.subplots()
    # PredictionErrorDisplay.from_predictions(y_test, pred, ax=ax)
    # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    # ax.set_title('Prediction Error Plot')
    # ax.set_xlabel('True Values')
    # ax.set_ylabel('Predicted Values')
    # plt.show()

    return results

def save_results_to_file(results, file_path, location, model_name):
    # Extract mean absolute error and format it with 3 decimal places
    mae_formatted = f"{results['mae']:.3f}"
    filename = f"{location}/{model_name}_{mae_formatted}.txt"
    full_path = os.path.join(file_path, filename)

    # Convert results to a human-readable string
    results_str = json.dumps(results, indent=4)

    # Save to a file
    with open(full_path, 'w') as file:
        file.write(results_str)

    print(f"Results saved to {full_path}")



def main(obs_data_path, est_data_path):

    # Read the data
    obs_X = pd.read_parquet(obs_data_path)
    est_X = pd.read_parquet(est_data_path)

    if 'date_forecast' in obs_X.columns:
        obs_X = obs_X.drop(columns='date_forecast')
    
    if 'date_forecast' in est_X.columns:
        est_X = est_X.drop(columns='date_forecast')

    if 'date_calc' in est_X.columns:
        est_X = est_X.drop(columns='date_calc')

    # Extract location from obs_data_path
    location = obs_data_path.split('_')[-1][0]

    # Test on models

    # XGBoost
    params = {
        'colsample_bytree': 0.8, 
        'gamma': 0.8, 
        'learning_rate': 0.008, 
        'max_depth': 10, 
        'min_child_weight': 10, 
        'n_estimators': 550, 
        'reg_alpha': 1, 
        'reg_lambda': 3, 
        'subsample': 0.912,
        'random_state': 0, 
        'booster': 'gbtree',
        'eval_metric': 'mae',
        'n_jobs': -1
    }

    results_xgb = test_xgboost(params, obs_X, est_X)
    
    # Save results
    save_results_to_file(results_xgb, 'preprocessing/test_results/xgb', location, 'xgb')

if __name__ == '__main__':
    obs_data_path = sys.argv[1]
    est_data_path = sys.argv[2]
    main(obs_data_path, est_data_path)
