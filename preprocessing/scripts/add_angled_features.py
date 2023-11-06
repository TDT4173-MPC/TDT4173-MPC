import pandas as pd
import numpy as np
import sys

def add_angled_features(df):
    """
    Experimental feature engineering.
    """

    angles = [0,10,25,35,50,75,85,100]


    for angle in angles:  # Loop over each angle
        # Use 'df' instead of 'A' to access the elements
        sun_elevation_interaction = np.cos(np.radians(angle - df['sun_elevation:d'])) * df['direct_rad:W']
        sun_azimuth_interaction = np.cos(np.radians(df['sun_azimuth:d'] - 180)) * df['direct_rad:W']
        cloud_cover_interaction = (1 - df['effective_cloud_cover:p']/100) * df['direct_rad:W']

        # Combine interactions into a composite feature
        composite_feature = sun_elevation_interaction + sun_azimuth_interaction + cloud_cover_interaction

        # Assign the composite feature to the current angle key in the df
        df[f'sun_elevation_interaction_{angle}'] = sun_elevation_interaction
        df[f'sun_azimuth_interaction_{angle}'] = sun_azimuth_interaction
        df[f'cloud_cover_interaction_{angle}'] = cloud_cover_interaction
        df[f'angle_radiation_{angle}'] = composite_feature
        

    df = df.dropna()
    df.replace([np.inf, -np.inf], 0, inplace=True)
    # Daylight Features (assuming 'date_forecast' and 'date_calc' are in datetime format)
    # df['daylight_duration'] = (df['date_forecast'] - df['date_calc']).dt.total_seconds() / 3600

    
    return df


def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    add_angled_features(df)

    # Save the modified data back to the same file
    df.to_parquet(input_file, index=False)
    print(f"angled feature added. File saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)