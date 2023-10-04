import pandas as pd
import sys

def keep_columns(df, columns):
    # Only keep columns that exist in the dataframe
    columns_to_keep = [col for col in columns if col in df.columns]
    df = df[columns_to_keep]
    return df


def main(columns_string, file_path):
    # Convert columns to list
    columns_to_keep = columns_string.split()

    # Read the data
    df = pd.read_parquet(file_path)
    
    # Keep columns
    df = keep_columns(df, columns_to_keep)
    
    # Save back to the same path
    df.to_parquet(file_path, index=False)
    print(f"Processed {file_path}")

if __name__ == '__main__':
    columns_string = sys.argv[1]
    input_file_path = sys.argv[2]
    main(columns_string, input_file_path)
