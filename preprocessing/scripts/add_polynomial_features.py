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


def main(input_file=0):

    # Read the data
    df = pd.read_parquet(input_file)

    df = add_polynomials(df, 2)

    # Save the modified data back to the same file
    df.to_parquet(input_file, index=False)
    print(f"Polynomial terms added. File saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)


