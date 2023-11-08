import pandas as pd
import sys

# Top 20 features for each set, hardcoded for simplicity
TOP_FEATURES_A = [
    "direct_rad:W", "diffuse_rad:W", "direct_rad:W_rate_of_change",
    "clear_sky_rad:W", "date_forecast_fft_amplitude", 
    "clear_sky_rad:W_rate_of_change_of_change", "clear_sky_rad:W_rate_of_change",
    "sun_azimuth:d", "direct_rad:W_rate_of_change_of_change", "diffuse_rad_1h:J",
    "t_1000hPa:K", "precip_5min:mm", "msl_pressure:hPa", "sun_elevation:d",
    "sun_elevation:d_fft_phase", "t_1000hPa:K_rate_of_change", "fresh_snow_24h:cm",
    "diffuse_rad:W_rate_of_change", "direct_rad_1h:J", "absolute_humidity_2m:gm3"
]

TOP_FEATURES_B = [
    "date_forecast_fft_phase", "direct_rad:W", "diffuse_rad:W", 
    "sun_elevation:d", "clear_sky_rad:W", "clear_sky_rad:W_rate_of_change", 
    "date_forecast_fft_amplitude", "cloud_base_agl:m", "year", "t_1000hPa:K", 
    "snow_drift:idx_fft_amplitude", "air_density_2m:kgm3", "diffuse_rad:W_rate_of_change", 
    "clear_sky_rad:W_rate_of_change_of_change", "t_1000hPa:K_rate_of_change", 
    "month", "diffuse_rad_1h:J", "direct_rad:W_rate_of_change", "visibility:m", 
    "precip_5min:mm"
]

TOP_FEATURES_C = [
    "direct_rad:W", "sun_elevation:d", "diffuse_rad:W", "t_1000hPa:K", 
    "direct_rad_1h:J", "date_forecast_fft_amplitude", "clear_sky_rad:W", 
    "clear_sky_energy_1h:J", "direct_rad:W_rate_of_change_of_change", 
    "snow_melt_10min:mm", "direct_rad:W_rate_of_change", "precip_5min:mm", 
    "relative_humidity_1000hPa:p", "msl_pressure:hPa", "precip_type_5min:idx_fft_amplitude", 
    "wind_speed_u_10m:ms", "diffuse_rad_1h:J", "sfc_pressure:hPa", 
    "dew_point_2m:K", "effective_cloud_cover:p"
]


# Add more top feature sets for other categories if necessary

def keep_top_features(df, top_features):
    # Only keep columns that are in the top features list
    return df[top_features]

def main(feature_set_label, file_path):
    # Map the feature set label to the corresponding list
    feature_sets = {
        'A': TOP_FEATURES_A,
        'B': TOP_FEATURES_B,
        'C': TOP_FEATURES_C,
        # Add additional mappings if there are more feature sets
    }
    
    top_features = feature_sets.get(feature_set_label.upper())

    if not top_features:
        raise ValueError(f"No top features defined for label: {feature_set_label}")

    # Read the data
    df = pd.read_parquet(file_path)
    
    # Keep top features
    df = keep_top_features(df, top_features)
    
    # Save back to the same path
    df.to_parquet(file_path, index=False)
    print(f"Processed {file_path} keeping top features for set {feature_set_label}")

if __name__ == '__main__':
    feature_set_label = sys.argv[1]  # Expecting 'A', 'B', 'C', etc.
    input_file_path = sys.argv[2]
    main(feature_set_label, input_file_path)

