from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import sys

def bin_columns(dataframe, columns_to_bin, n_bins=5):
    """
    Bins the specified columns of the dataframe into equal-sized bins.
    
    Parameters:
    - dataframe: pd.DataFrame
    - columns_to_bin: list of strings, the names of the columns to bin
    - n_bins: int or dict, the number of bins for each column (if int, use the same number for all columns;
              if dict, specify individual numbers with column names as keys)
    
    Returns:
    - binned_dataframe: pd.DataFrame, the dataframe with the specified columns binned
    """
    binned_dataframe = dataframe.copy()
    
    for column in columns_to_bin:
        # Determine the number of bins for this column
        bins = n_bins if isinstance(n_bins, int) else n_bins.get(column, 5)
        
        # Create quantile-based bins
        binned_dataframe[f'binned_{column}'] = pd.qcut(
            binned_dataframe[column],
            q=bins,
            labels=False,
            duplicates='drop'
        )
        
    return binned_dataframe


def main(input_file=0):

    # Read the data
    df = pd.read_parquet(input_file)

    # Specify the columns to bin
    columns_to_bin = [
        'super_cooled_liquid_water:kgm2',
        'ceiling_height_agl:m',
        'cloud_base_agl:m'
    ]

    # Bin the columns
    # df = bin_columns(df, columns_to_bin)
    df = bin_columns(df, ['effective_cloud_cover:p'], n_bins=2)
    df = bin_columns(df, ['ceiling_height_agl:m'], n_bins=3)
    df = bin_columns(df, ['average_wind_speed'], n_bins=5)

    df.to_parquet(input_file, index=False)
    print(f"Features binned. File saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)


