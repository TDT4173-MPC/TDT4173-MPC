import pandas as pd
import sys

def add_lagged_features(df, features, max_lag=24, fill_value=None):
    """
    Adds lagged columns for specified features in the dataframe.
    Assumes the dataframe is time sorted. 'max_lag' determines how many lags to create.
    'fill_value' is what to fill the NaNs with, after shifting.
    """
    for lag in range(1, max_lag + 1):
        for feature in features:
            lag_column_name = f"{feature}_lag_{lag}"
            df[lag_column_name] = df[feature].shift(lag).fillna(fill_value) # Shifts the column downwards and fills NaNs if required
    return df

def main(input_file, max_lag=24):
    # Define the features for which to calculate lagged features
    features_to_lag = [
        "diffuse_rad:W",
        "direct_rad:W",
        "effective_cloud_cover:p",
        "fresh_snow_24h:cm",
        "sun_elevation:d",
        "absolute_humidity_2m:gm3",
        "super_cooled_liquid_water:kgm2",
        "t_1000hPa:K",
        "total_cloud_cover:p",
        "air_density_2m:kgm3",
        "clear_sky_rad:W",
        "visibility:m",
        "relative_humidity_1000hPa:p",
        "msl_pressure:hPa",
        "snow_water:kgm2",
        "dew_point_2m:K",
        "wind_speed_u_10m:ms",
        "direct_rad_1h:J",
        "diffuse_rad_1h:J",
        "clear_sky_energy_1h:J",
        "wind_speed_10m:ms",
        "wind_speed_v_10m:ms",
        "elevation:m",
        "precip_5min:mm",
        "is_day:idx",
        "is_in_shadow:idx",
        "precip_type_5min:idx",
        "pressure_100m:hPa",
        "pressure_50m:hPa",
        "rain_water:kgm2",
        "sfc_pressure:hPa",
        "snow_depth:cm",
        "snow_melt_10min:mm",
        "sun_azimuth:d",
        "ceiling_height_agl:m",
        "cloud_base_agl:m",
        "prob_rime:p",
        "dew_or_rime:idx",
        "fresh_snow_3h:cm",
        "snow_density:kgm3",
        "fresh_snow_6h:cm",
        "fresh_snow_12h:cm",
        "fresh_snow_1h:cm",
        "wind_speed_w_1000hPa:ms",
        "snow_drift:idx"
    ]

    # Read the data
    df = pd.read_parquet(input_file)

    # Add lagged features
    df_with_lags = add_lagged_features(df, features_to_lag, max_lag, fill_value=0) # You can change the fill_value as needed

    # Save the modified data back to the same path
    df_with_lags.to_parquet(input_file, index=False)
    print(f"Lagged features up to {max_lag} added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    max_lag_periods = int(sys.argv[2]) # The second argument passed to the script is the max_lag value
    main(input_file_path, max_lag_periods)
