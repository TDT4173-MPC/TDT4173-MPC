import pandas as pd
import sys

def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Remove the rows where the target is nan
    try:
        df = df[df['pv_measurement'].notna()]
    except KeyError:
        pass

    # Fill NaNs with forward fill
    df_ffill = df.fillna(method='ffill')
    
    # Fill remaining NaNs with backward fill
    df_bfill = df_ffill.fillna(method='bfill')

    # Interpolate the remaining NaNs
    df_interpolated = df_bfill.interpolate(method='linear')

    # Replace NaNs with rolling window mean (you can also use median)
    window_size = 3  # Change the window size as per your dataset
    df_rolling = df_interpolated.fillna(df_interpolated.rolling(window=window_size, min_periods=1).mean())

    # Save the handled data back to the same path
    df_rolling.to_parquet(input_file, index=False)
    print(f"NaNs handled in {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
