import pandas as pd
import numpy as np
import sys

def add_cosines(df, column, period, num_terms):
    """
    Adds cosine terms for a specific column to capture seasonality.
    """
    for i in range(1, num_terms + 1):
        cosine_name = f"{column}_cosine_{i}"
        df[cosine_name] = np.cos((2 * np.pi * i * df[column]) / period)
    
    return df


def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Assuming your df has 'month', 'day', and 'hour' columns (numeric) representing the respective time parts.
    # Define the periods of your seasonal cycles.
    month_period = 12
    day_period = 365.25  # accounts for leap years
    hour_period = 24

    # Add Fourier terms for seasonal patterns captured by month, day, and hour
    df = add_cosines(df, 'month', month_period, num_terms=2)
    df = add_cosines(df, 'day', day_period, num_terms=2)
    df = add_cosines(df, 'hour', hour_period, num_terms=2)

    # Save the modified data back to the same file
    df.to_parquet(input_file, index=False)
    print(f"Fourier terms added. File saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
