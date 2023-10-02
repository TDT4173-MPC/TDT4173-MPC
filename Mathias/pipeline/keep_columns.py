import pandas as pd
import sys

def keep_columns(df, columns):
    df = df[columns]
    return df

def main(columns_string, file_path):
    # Convert columns to list
    columns_to_keep = columns_string.split()

    print(f"Keeping columns: {columns_to_keep}")

    # Read the data
    df = pd.read_csv(file_path)
    
    # Keep columns
    df = keep_columns(df, columns_to_keep)
    
    # Save back to the same path
    df.to_csv(file_path, index=False)
    print(f"Processed {file_path}")

if __name__ == '__main__':
    columns_string = sys.argv[1]
    input_file_path = sys.argv[2]
    main(columns_string, input_file_path)
