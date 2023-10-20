import pandas as pd
import matplotlib.pyplot as plt

def plot_columns(df, column_x, column_y):

    # Check if the columns exist in the DataFrame
    if column_x not in df.columns or column_y not in df.columns:
        raise ValueError(f"One or both of the columns: '{column_x}' or '{column_y}' do not exist in the DataFrame.")

    # Create a scatter plot of the two columns
    plt.figure(figsize=(10,6))  # You can adjust the size as per your need
    plt.scatter(df[column_x], df[column_y], alpha=0.5)

    # Adding a title and labels
    plt.title(f'Scatter plot of {column_x} vs {column_y}')
    plt.xlabel(column_x)
    plt.ylabel(column_y)

    # Optional: Include a grid
    plt.grid(True)

    # Show the plot
    plt.show()


# Import data
# Read in the data
data_path = './preprocessing/data'
obs_A = pd.read_parquet(f'{data_path}/obs_A.parquet').drop(columns='date_forecast')
est_A = pd.read_parquet(f'{data_path}/est_A.parquet').drop(columns='date_forecast')
obs_B = pd.read_parquet(f'{data_path}/obs_B.parquet').drop(columns='date_forecast')
est_B = pd.read_parquet(f'{data_path}/est_B.parquet').drop(columns='date_forecast')
obs_C = pd.read_parquet(f'{data_path}/obs_C.parquet').drop(columns='date_forecast')
est_C = pd.read_parquet(f'{data_path}/est_C.parquet').drop(columns='date_forecast')

# Concatenate
A = pd.concat([obs_A, est_A])
B = pd.concat([obs_B, est_B])
C = pd.concat([obs_C, est_C])

try:
    plot_columns(B, 'sun_elevation:d', 'pv_measurement')
except Exception as e:
    print(e)
