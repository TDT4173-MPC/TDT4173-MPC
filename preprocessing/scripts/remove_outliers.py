import pandas as pd
import sys

def remove_outliers(df):
    """
    Remove outliers from a DataFrame using the IQR method.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers for numeric columns only
    for col in numeric_cols:
        df = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])]

    return df


def main(input_file, output_file):
    # Read the data
    df = pd.read_csv(input_file)

    # Remove outliers
    df_no_outliers = remove_outliers(df)

    # Save the cleaned data
    df_no_outliers.to_csv(output_file, index=False)
    print(f"Outliers removed. Cleaned data saved to {output_file}.")


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = input_file_path
    main(input_file_path, output_file_path)
