import pandas as pd
import sys

def remove_constant_regions(dataframe, column_name="pv_measurement", threshold=72):
    """
    Removes rows where the specified column has constant values for more than the given threshold.
    """
    
    # Check if the specified column exists in the dataframe
    if column_name not in dataframe.columns:
        print(f"Warning: '{column_name}' column not found in the dataframe. No rows removed.")
        return dataframe
    
    same_as_previous = dataframe[column_name].eq(dataframe[column_name].shift())
    group_ids = (~same_as_previous).cumsum()
    to_remove = group_ids[same_as_previous].value_counts() > threshold
    group_ids_to_remove = to_remove[to_remove].index
    
    # Drop entire rows that match the conditions
    return dataframe.drop(dataframe[group_ids.isin(group_ids_to_remove)].index)



def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)
    
    # Remove constant regions
    df = remove_constant_regions(df)

    # Save the modified data back to the same path
    df.to_parquet(input_file, index=False)
    print(f"Date features added, constant regions removed, and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
