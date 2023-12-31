import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

def test_feature(df):
    """
    Experimental feature engineering.
    """

    # Radiation Features
    df['total_radiation:W'] = df['direct_rad:W'] + df['diffuse_rad:W']
    df['total_radiation_1h:J'] = df['direct_rad_1h:J'] + df['diffuse_rad_1h:J']
    df['rad_diff:W'] = df['direct_rad:W'] - df['diffuse_rad:W']
    df['rad_diff_1h:J'] = df['direct_rad_1h:J'] - df['diffuse_rad_1h:J']
    df['diffuse_direct_ratio'] = df['diffuse_rad:W'] / df['direct_rad:W']
    # df['total_radiation_sun_azimuth_interaction'] = df['total_radiation:W'] * df['sun_azimuth:d']
    # df['total_radiation_sun_elevation_interaction'] = df['total_radiation:W'] * df['sun_elevation:d']
    # df['direct_rad_ratio'] = df['direct_rad_1h:J'] / df['direct_rad:W']
    # df['diffuse_rad_ratio'] = df['diffuse_rad_1h:J'] / df['diffuse_rad:W']
    # df['clear_sky_ratio'] = df['clear_sky_rad:W'] / df['total_radiation:W']

    # Temperature and Pressure Features
    df['temp_dewpoint_diff'] = df['t_1000hPa:K'] - df['dew_point_2m:K']
    df['pressure_gradient'] = df['pressure_100m:hPa'] - df['pressure_50m:hPa']
    df['t_1000hPa:C'] = df['t_1000hPa:K'] - 273.15
    df['dew_point_2m:C'] = df['dew_point_2m:K'] - 273.15
    df['msl_pressure:hPa_scaled'] = MinMaxScaler().fit_transform(df['msl_pressure:hPa'].values.reshape(-1, 1))
    df['sfc_pressure:hPa_scaled'] = MinMaxScaler().fit_transform(df['sfc_pressure:hPa'].values.reshape(-1, 1))

    # Wind Features
    df['wind_vector_magnitude'] = (df['wind_speed_u_10m:ms']**2 + df['wind_speed_v_10m:ms']**2 + df['wind_speed_w_1000hPa:ms']**2)**0.5
    df['average_wind_speed'] = (df['wind_speed_10m:ms'] + df['wind_speed_u_10m:ms']) / 2

    # Cloud and Snow Features
    df['cloud_humidity_product'] = df['total_cloud_cover:p'] * df['absolute_humidity_2m:gm3']
    df['snow_accumulation'] = df[['fresh_snow_24h:cm', 'fresh_snow_12h:cm', 'fresh_snow_6h:cm', 'fresh_snow_3h:cm', 'fresh_snow_1h:cm']].sum(axis=1)
    # Humidity Features
    # df['humidity_ratio'] = df['absolute_humidity_2m:gm3'] / df['relative_humidity_1000hPa:p']

    # Interaction between radiation and cloud cover
    df['radiation_cloud_interaction'] = df['direct_rad:W'] * df['effective_cloud_cover:p']

    # Interaction between temperature and radiation (considering that high temperature may reduce efficiency)
    df['temp_rad_interaction'] = df['t_1000hPa:K'] * df['total_radiation:W']

    # Interaction between wind cooling effect and temperature
    df['wind_temp_interaction'] = df['average_wind_speed'] * df['t_1000hPa:K']

    # Interaction between humidity and temperature
    df['humidity_temp_interaction'] = df['absolute_humidity_2m:gm3'] * df['t_1000hPa:K']

    # Interaction between humidity and radiation
    df['sun_elevation_direct_rad_interaction'] = df['sun_elevation:d'] * df['direct_rad:W']

    df['precip'] = df['precip_5min:mm']*df['precip_type_5min:idx']

    # df = df.dropna()
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
