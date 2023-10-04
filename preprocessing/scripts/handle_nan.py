import pandas as pd
import sys

def handle_nan(df):
    """
    Setting all nan values to zero in a DataFrame. And removing nan targets.
    """
    return df.fillna(0)

def main(input_file):
    # Read the data
    df = pd.read_csv(input_file)

    # Remove the rows where target is nan
    df = df[df['pv_measurement'].notna()]

    # Set all remaining nans to 0
    df_handled = handle_nan(df)

    # Save the handled data back to the same path
    df_handled.to_csv(input_file, index=False)
    print(f"NaNs set to zero in {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
