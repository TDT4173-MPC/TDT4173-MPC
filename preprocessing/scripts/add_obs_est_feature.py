import pandas as pd
import sys

def add_est_obs_feature(df):
    """
    Adds a column to the dataframe that indicates whether the data is estimated or observed.
    """
    # Add the est_obs feature
    if 'date_calc' not in df.columns:
        # If 'date_calc' does not exist, create 'observed' column and set to 1
        df['observed'] = 1
        return df
    else:
        # If 'date_calc' exists, create a new column and set values to 0
        df['observed'] = 0
        return df.drop(columns=['date_calc'])

    return df

def main(input_file):

    # Read the data
    df = pd.read_parquet(input_file)

    # Add rate of change features
    df_with_est_obs = add_est_obs_feature(df)

    # Save the modified data back to the same path
    df_with_est_obs.to_parquet(input_file, index=False)
    print(f"Est obs feature created and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
