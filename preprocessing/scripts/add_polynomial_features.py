from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import sys

def add_polynomials(df, degree):
    """
    Adds polynomial terms.
    """
    try:
        time_features = df[['date_forecast', 'date_calc']]
        df = df.drop(columns=['date_forecast', 'date_calc'])
    except:
        time_features = df['date_forecast']
        df = df.drop(columns='date_forecast')

    df = pd.DataFrame(PolynomialFeatures(degree, include_bias=False).fit_transform(df))
    df = pd.concat([df, time_features], axis=1)

    return df


def multiply_and_add_columns(df, column_to_multiply, excluded_column):
    for col in df.columns:
        if col not in [column_to_multiply, excluded_column]:
            # Multiply and create a new column
            try:
                new_column_name = f"{col}_multiplied_by_{column_to_multiply}"
                df[new_column_name] = df[col] * df[column_to_multiply]
            except:
                pass
    return df

def multiply_and_add_single_column(df, column_1, column_2, normalize=False):
    df[f"{column_1}_multiplied_by_{column_2}"] = df[column_1] * df[column_2]
    if normalize:
        df[f"{column_1}_multiplied_by_{column_2}"] = (df[f"{column_1}_multiplied_by_{column_2}"] - df[f"{column_1}_multiplied_by_{column_2}"].mean()) / df[f"{column_1}_multiplied_by_{column_2}"].std()
    return df




def main(input_file=0):

    # Read the data
    df = pd.read_parquet(input_file)

    # df = add_polynomials(df, 2)

    # For trying out big chunks of interactions
    # df = multiply_and_add_columns(df, 'effective_cloud_cover:p', 'pv_measurement')
    # df = multiply_and_add_columns(df, 'sun_elevation:d', 'pv_measurement')

    # For trying out single interactions
    # df = multiply_and_add_single_column(df, 'clear_sky_rad:W', 'sun_elevation:d', normalize=True)
    df = multiply_and_add_single_column(df, 'absolute_humidity_2m:gm3', 'effective_cloud_cover:p')
    df = multiply_and_add_single_column(df, 'hour', 'effective_cloud_cover:p')

    # Save the modified data back to the same file
    df.to_parquet(input_file, index=False)
    print(f"Polynomial terms added. File saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)


