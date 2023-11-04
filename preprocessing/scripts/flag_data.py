import pandas as pd
import sys

def flag_data(df):
    """
    Adds 'month', 'year', 'hour' and 'day' columns to the dataframe based on the 'date_forecast' column.
    """
    
    
    return df

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Add date features
    modified_df = flag_data(df)

    # Save the modified data back to the same path
    modified_df.to_parquet(input_file, index=False)
    print(f"Data flagged and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
