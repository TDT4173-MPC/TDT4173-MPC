import itertools
import statsmodels.api as sm
import pandas as pd
from multiprocessing import Pool, cpu_count

#Importing data:
data_path = './data/A'
target_A = pd.read_parquet(f'{data_path}/train_targets.parquet')
test_A = pd.read_parquet(f'{data_path}/X_test_estimated.parquet')

#TODO: Give the triaining model only one year of data


import numpy as np
import itertools


# Assuming you have loaded your data into a DataFrame named df
# For example:
# df = pd.read_csv("your_data.csv")
# Ensure the 'time' column is a datetime type and set it as the index
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)

def sarima_grid_search(params):
    p, d, q, P, D, Q, s = params
    order = (p, d, q)
    seasonal_order = (P, D, Q, s)
    
    try:
        model = sm.tsa.statespace.SARIMAX(target_A['pv_measurement'],
                                          order=order,
                                          seasonal_order=seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        results = model.fit(disp=False)
        return results.aic, order, seasonal_order
    except:
        return float('inf'), order, seasonal_order

def main():
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 5)

    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, d and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in pdq]

    # List with all possible combinations of p, d, q and P, D, Q
    grid_params = list(itertools.product(p, d, q, p, d, q))
    grid_params = [(x[0], x[1], x[2], x[3], x[4], x[5], 24) for x in grid_params]

    # Use the Pool class to parallelize the computation
    pool = Pool(cpu_count())

    results = pool.map(sarima_grid_search, grid_params)

    pool.close()
    pool.join()

    # Extract the order and seasonal_order with the lowest AIC
    min_aic = min([res[0] for res in results])
    best_order = [res[1] for res in results if res[0] == min_aic][0]
    best_seasonal_order = [res[2] for res in results if res[0] == min_aic][0]

    print(f"Best AIC: {min_aic}")
    print(f"Best order: {best_order}")
    print(f"Best seasonal order: {best_seasonal_order}")

if __name__ == '__main__':
    main()
