import pandas as pd
import sys

def keep_columns(df, columns):
    df = df[columns]
    return df

def main(columns_string):

    # Convert columns to list
    columns_to_keep = columns_string.split()

    print(f"Keeping columns: {columns_to_keep}")

    # Read in the data
    x_obs_A = pd.read_csv('./Mathias/pipeline/data/x_obs_A.csv')
    x_est_A = pd.read_csv('./Mathias/pipeline/data/x_est_A.csv')

    x_obs_B = pd.read_csv('./Mathias/pipeline/data/x_obs_B.csv')
    x_est_B = pd.read_csv('./Mathias/pipeline/data/x_est_B.csv')

    x_obs_C = pd.read_csv('./Mathias/pipeline/data/x_obs_C.csv')
    x_est_C = pd.read_csv('./Mathias/pipeline/data/x_est_C.csv')

    # Keep columns
    x_obs_A = keep_columns(x_obs_A, columns_to_keep)
    x_est_A = keep_columns(x_est_A, columns_to_keep)

    x_obs_B = keep_columns(x_obs_B, columns_to_keep)
    x_est_B = keep_columns(x_est_B, columns_to_keep)

    x_obs_C = keep_columns(x_obs_C, columns_to_keep)
    x_est_C = keep_columns(x_est_C, columns_to_keep)

    # Save
    x_obs_A.to_csv('./Mathias/pipeline/data/x_obs_A.csv')
    x_est_A.to_csv('./Mathias/pipeline/data/x_est_A.csv')

    x_obs_B.to_csv('./Mathias/pipeline/data/x_obs_B.csv')
    x_est_B.to_csv('./Mathias/pipeline/data/x_est_B.csv')

    x_obs_C.to_csv('./Mathias/pipeline/data/x_obs_C.csv')
    x_est_C.to_csv('./Mathias/pipeline/data/x_est_C.csv')


if __name__ == '__main__':
    columns_string = sys.argv[1]
    main(columns_string)
