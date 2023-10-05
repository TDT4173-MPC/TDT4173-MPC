import pandas as pd
import sys

def add_date_features(df):
    """
    Adds 'month', 'year', and 'time_of_day' columns to the dataframe based on the 'date_forecast' column.
    """
    
    # Check if 'date_forecast' exists in the dataframe
    if 'date_forecast' in df.columns:
        # Convert the 'date_forecast' column to datetime format
        df['date_forecast'] = pd.to_datetime(df['date_forecast'])
        
        # Extract month, year, and time of day
        df['month'] = df['date_forecast'].dt.month
        df['year'] = df['date_forecast'].dt.year
        
        # Convert time to the number of minutes since midnight
        df['time_of_day'] = df['date_forecast'].dt.hour * 60 + df['date_forecast'].dt.minute
    else:
        print("Warning: 'date_forecast' column not found in the dataframe. No date features added.")
        return df
    
    return df.drop(columns=['date_forecast'])

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
