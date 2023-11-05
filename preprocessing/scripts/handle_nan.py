import pandas as pd
import sys

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Remove the rows where target is nan
    try:
        df = df[df['pv_measurement'].notna()]
    except KeyError:
        pass

    # Set all remaining nans to 0
    df_handled = df.fillna(0)

    # Save the handled data back to the same path
    df_handled.to_parquet(input_file, index=False)
    print(f"NaNs set to zero in {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
