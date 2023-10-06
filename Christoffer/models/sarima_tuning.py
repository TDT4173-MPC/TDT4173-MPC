import statsmodels.api as sm
import numpy as np
import pandas as pd
import itertools


#Importing data:
data_path = './data/A'
target_A = pd.read_parquet(f'{data_path}/train_targets.parquet')
test_A = pd.read_parquet(f'{data_path}/X_test_estimated.parquet')

p = d = q = range(0, 10)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]  # assuming hourly data with a daily seasonality

min_aic = np.inf
best_order = None
best_seasonal_order = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(target_A,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            if results.aic < min_aic:
                min_aic = results.aic
                best_order = param
                best_seasonal_order = param_seasonal
        except:
            continue

print(f"Best order: {best_order}")
print(f"Best seasonal order: {best_seasonal_order}")