import pandas as pd
import sys

import pandas as pd

def local_winsorize(df, column_name="pv_measurement", window_size=100, lower_quantile=0.05, upper_quantile=0.98):
    # Add a new column for winsorized measurements or create a new one if it doesn't exist
    if 'winsorized' not in df.columns:
        df['winsorized'] = df[column_name]

    # Apply winsorization using a rolling window
    for start in range(0, len(df)):
        end = min(start + window_size, len(df)) # Ensure the end does not go beyond the DataFrame
        window = df[column_name].iloc[start:end]
        lower_bound = window.quantile(lower_quantile)
        upper_bound = window.quantile(upper_quantile)
        df.loc[start:end-1, 'winsorized'] = window.clip(lower=lower_bound, upper=upper_bound)
    
    # switch pv_measurement with winsorized values
    df[column_name] = df['winsorized']
    df = df.drop(columns=['winsorized'])

    return df



def main(input_file, window_size=100, lower_quantile=0.05, upper_quantile=0.98):
    # Read the data
    df = pd.read_parquet(input_file)
    
    # Apply local winsorization
    df = local_winsorize(df, window_size=window_size, lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    # Save the modified data back to the same path
    df.to_parquet(input_file, index=False)
    print(f"Winsorized data with window size {window_size} and saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]  # The first argument passed to the script is the input file path
    main(input_file_path)
