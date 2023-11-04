import pandas as pd
import numpy as np
import sys

def add_date_features(df):
    """
    Adds 'month', 'year', 'hour' and 'day' columns to the dataframe based on the 'date_forecast' column.
    """
    
    # Check if 'date_forecast' exists in the dataframe
    if 'date_forecast' in df.columns:
        # Convert the 'date_forecast' column to datetime format
        df['date_forecast'] = pd.to_datetime(df['date_forecast'])
        
        # Extract month, year, hour and day
        df['month'] = df['date_forecast'].dt.month
        df['year'] = df['date_forecast'].dt.year
        df['hour'] = df['date_forecast'].dt.hour
        # df['day'] = df['date_forecast'].dt.day

    else:
        print("Warning: 'date_forecast' column not found in the dataframe. No date features added.")
        return df
    
    return df

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Add date features
    modified_df = add_date_features(df)

    # Save the modified data back to the same path
    modified_df.to_parquet(input_file, index=False)
    print(f"Date features added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
