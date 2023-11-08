import pandas as pd
import os

# Function to calculate correlation and update the features info
def update_feature_correlations(parquet_file_directory, features_info_file):
    # Check if the directory exists
    if not os.path.exists(parquet_file_directory):
        print(f"Directory does not exist: {parquet_file_directory}")
        return
    
    # Check if the features_info CSV file exists
    if not os.path.isfile(features_info_file):
        print(f"File does not exist: {features_info_file}")
        return
    
    # Load your existing features information
    features_info_df = pd.read_csv(features_info_file)

    # DataFrame for collected data
    combined_dataframe = pd.DataFrame()

    # Read each parquet file in the specified directory
    for entry in os.listdir(parquet_file_directory):
        if entry.endswith('.parquet'):
            file_path = os.path.join(parquet_file_directory, entry)
            data = pd.read_parquet(file_path)
            
            # Combine data from all parquet files (if there are multiple)
            if combined_dataframe.empty:
                combined_dataframe = data
            else:
                combined_dataframe = pd.concat([combined_dataframe, data])

    # Drop rows with missing target values
    combined_dataframe = combined_dataframe.dropna(subset=['pv_measurement'])

    # Calculate correlations with the target for each feature
    correlations = combined_dataframe.corr()['pv_measurement'].drop('pv_measurement')

    # Update the 'correlation' column with the new values
    features_info_df['correlation'] = features_info_df['feature'].map(correlations)

    # Save the updated DataFrame back to CSV
    features_info_df.to_csv(features_info_file, index=False)

    print(f"Updated feature correlations in {features_info_file}")

# Usage of the function with appropriate file path
parquet_dir = 'path/to/your/parquet_files'  # replace with your parquet files directory
features_csv = 'features_info.csv'  # assuming the CSV file is in the current working directory
update_feature_correlations(parquet_dir, features_csv)
