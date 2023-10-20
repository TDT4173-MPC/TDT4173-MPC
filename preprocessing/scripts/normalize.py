import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler


def normalize(df, columns):
    """
    Normalizes the data using the activation function.
    """
    scaler = StandardScaler()
    for column in columns:
        df[column] = scaler.fit_transform(df[[column]])
    return df


def main(input_file, output_file, columns_to_normalize):
    # Read the data
    df = pd.read_parquet(input_file)

    # Normalize specified columns
    df_modified = normalize(df, columns_to_normalize)

    # Save the Normalized data
    df_modified.to_parquet(output_file, index=False)
    print(f"Files modified. Normalized data saved to {output_file}.")


if __name__ == "__main__":
    file_path = sys.argv[1]

    # You can specify the columns you want to normalize here or pass through the command line
    columns_to_normalize = ['direct_rad:W']  # Example column names to be normalized

    if len(sys.argv) > 3:
        columns_to_normalize = sys.argv[3:]

    main(file_path, file_path, columns_to_normalize)
