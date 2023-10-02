import pandas as pd
import sys

def interpolate_missing_values(df):
    """
    Interpolate missing values (NaNs) in a DataFrame.
    """
    return df.interpolate()

def main(input_file):
    # Read the data
    df = pd.read_csv(input_file)

    # Interpolate the missing values
    df_interpolated = interpolate_missing_values(df)

    # Save the interpolated data back to the same path
    df_interpolated.to_csv(input_file, index=False)
    print(f"Missing values interpolated and saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
