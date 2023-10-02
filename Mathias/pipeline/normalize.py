import pandas as pd
import sys

def normalize_data(df):
    """
    Normalize numeric columns of a DataFrame.
    """
    # Select only numeric columns (excluding datetime columns)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Normalize the numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())

    return df

def main(input_file):
    # Read the data
    df = pd.read_csv(input_file)

    # Normalize the data
    df_normalized = normalize_data(df)

    # Save the normalized data back to the same path
    df_normalized.to_csv(input_file, index=False)
    print(f"Data normalized and saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
