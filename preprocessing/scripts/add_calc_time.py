import pandas as pd
import sys

def add_calc_time(df):
    """
    Adds a 'calc_time_before' column to the dataframe.
    If the dataframe has a 'date_calc' column, the new column will contain the difference 
    in hours between 'date_forecast' and 'date_calc'. 
    If there's no 'date_calc' column, the new column will contain all zeros.
    """
    
    # Check if 'date_calc' and 'date_forecast' exist in the dataframe
    if 'date_calc' in df.columns and 'date_forecast' in df.columns:
        # Convert the columns to datetime format
        df['date_forecast'] = pd.to_datetime(df['date_forecast'])
        df['date_calc'] = pd.to_datetime(df['date_calc'])
        
        # Calculate the difference in hours
        df['calc_time_before'] = (df['date_forecast'] - df['date_calc']).dt.total_seconds() / 3600
        return df.drop(columns=['date_calc'])
    else:
        # If 'date_calc' doesn't exist, add 'calc_time_before' with all zeros
        df['calc_time_before'] = 0

    if 'date_calc' not in df.columns:
        return df
    else:
        return df.drop(columns=['date_calc'])


def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Add features
    modified_df = add_calc_time(df)

    # Save the normalized data back to the same path
    modified_df.to_parquet(input_file, index=False)
    print(f"Features added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
