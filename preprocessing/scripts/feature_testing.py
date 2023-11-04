import pandas as pd
import numpy as np
import sys

def test_feature(df):
    """
    Experimental feature engineering.
    """

    # Radiation Features
    df['total_radiation'] = df['direct_rad:W'] + df['diffuse_rad:W']
    df['direct_rad_ratio'] = df['direct_rad_1h:J'] / df['direct_rad:W']
    df['diffuse_rad_ratio'] = df['diffuse_rad_1h:J'] / df['diffuse_rad:W']
    df['clear_sky_ratio'] = df['clear_sky_rad:W'] / df['total_radiation']

    # Temperature and Pressure Features
    df['temp_dewpoint_diff'] = df['t_1000hPa:K'] - df['dew_point_2m:K']
    df['pressure_gradient'] = df['pressure_100m:hPa'] - df['pressure_50m:hPa']

    # Wind Features
    df['wind_vector_magnitude'] = (df['wind_speed_u_10m:ms']**2 + df['wind_speed_v_10m:ms']**2 + df['wind_speed_w_1000hPa:ms']**2)**0.5
    df['average_wind_speed'] = (df['wind_speed_10m:ms'] + df['wind_speed_u_10m:ms']) / 2
    df = df.drop(columns=['wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'wind_speed_10m:ms'])

    # Cloud and Snow Features
    df['cloud_thickness'] = df['ceiling_height_agl:m'] - df['cloud_base_agl:m']
    df['snow_accumulation'] = df[['fresh_snow_24h:cm', 'fresh_snow_12h:cm', 'fresh_snow_6h:cm', 'fresh_snow_3h:cm', 'fresh_snow_1h:cm']].sum(axis=1)
    df = df.drop(columns=['fresh_snow_24h:cm', 'fresh_snow_12h:cm', 'fresh_snow_6h:cm', 'fresh_snow_3h:cm', 'fresh_snow_1h:cm'])

    # Humidity Features
    df['humidity_ratio'] = df['absolute_humidity_2m:gm3'] / df['relative_humidity_1000hPa:p']

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    # Daylight Features (assuming 'date_forecast' and 'date_calc' are in datetime format)
    # df['daylight_duration'] = (df['date_forecast'] - df['date_calc']).dt.total_seconds() / 3600

    
    return df


def main(input_file):
    # Read the data
    df = pd.read_parquet(input_file)

    test_feature(df)

    # Save the modified data back to the same file
    df.to_parquet(input_file, index=False)
    print(f"Experimental feature added. File saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    main(input_file_path)
