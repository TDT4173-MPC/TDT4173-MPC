import pandas as pd
import sys

def normalize_pressure(df):
    """
    Subtracts mean value from pressure columns.
    """
    
    pressure_columns = [
        'msl_pressure:hPa',
        'pressure_100m:hPa',
        'pressure_50m:hPa',
        'sfc_pressure:hPa'
    ]

    # Only consider columns that exist in the dataframe
    existing_pressure_columns = [col for col in pressure_columns if col in df.columns]

    df[existing_pressure_columns] = df[existing_pressure_columns] - df[existing_pressure_columns].mean()

    return df


def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    # Normalize the pressure data
    df_normalized = normalize_pressure(df)

    # Save the normalized pressure data back to the same path
    df_normalized.to_parquet(input_file, index=False)
    print(f"Data normalized and saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
