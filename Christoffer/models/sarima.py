import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load your dataset 
data_path = './data/A'
#obs_A = pd.read_parquet(f'{data_path}/X_train_estimated.parquet')
#est_A = pd.read_parquet(f'{data_path}/X_train_observed.parquet')
target_A = pd.read_parquet(f'{data_path}/train_targets.parquet')

print(target_A.head(100))


# obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet')
# est_B = pd.read_parquet(f'{data_path}/est_B.parquet')
# obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet')
# est_C = pd.read_parquet(f'{data_path}/est_C.parquet')

test_A = pd.read_parquet(f'{data_path}/X_test_estimated.parquet')
# test_B = pd.read_parquet(f'{data_path}/test_B.parquet').dropna()
# test_C = pd.read_parquet(f'{data_path}/test_C.parquet').dropna()

# Concatenate
# A = pd.concat([obs_A, est_A])
# B = pd.concat([obs_B, est_B])
# C = pd.concat([obs_C, est_C])

#print(A.head(100))
# X_B = B.drop(columns=['pv_measurement'])
# y_B = B['pv_measurement']
# X_C = C.drop(columns=['pv_measurement'])
# y_C = C['pv_measurement']


#print(A.head)


#print(target_A.head(100))

# If you don't know the orders, you might want to visualize the data and its ACF/PACF first
# This will help you determine the ARIMA order
# plt.figure()
# target_A.plot()
# plt.title("Your Time Series Data")

# # plt.figure()
# # plot_acf(A)
# # plt.title("ACF")

# # plt.figure()
# # plot_pacf(A)
# # plt.title("PACF")

# plt.show()


# # # Decompose the data to see the trend, seasonal, and residual components
# # # This will help you determine the SARIMA order
# decomposition = seasonal_decompose(target_A, model="additive", period=24)
# decomposition.plot()
# plt.show()



# # # Let's assume you have determined the order and seasonal order (p,d,q,P,D,Q,s)
# # # For this example, I'll use random orders, but you should adjust based on your data's characteristics
order = (0,0,0)
seasonal_order = (0,0,0,24)  # Assuming daily seasonality for hourly data

# # # Split data into training and testing sets
# #train = data[:-24*7]  # Leaving the last week for testing
# #test = data[-24*7:]

# # # Fit the SARIMA model
# model = SARIMAX(target_A, order=order, seasonal_order=seasonal_order)
# results = model.fit(disp=True)

# # Get predictions
# predictions = results.forecasts(steps = 720)

# # Plot results
# plt.figure()
# plt.plot(train.index, train, label="Train")
# plt.plot(test.index, test, label="Test")
# plt.plot(test.index, predictions, label="Predicted")
# plt.legend(loc="best")
# plt.title("SARIMA Forecast")
# plt.show()

# You can further evaluate the model using metrics like MAE, RMSE, etc.
