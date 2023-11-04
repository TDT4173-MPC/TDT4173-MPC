import pandas as pd
from datetime import datetime

# Define your CSV headers and default values
csv_headers = [
    "feature", "description", "rf_score", "xgb_score", "correlation", 
    "nans", "performance", "comments", "update_time", "responsible_person", "status"
]

DEFAULT_VALUE = "N/A"  # Placeholder for fields to be updated later
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current date and time

# List all your features here
features = [
    "pv_measurement",
    "date_forecast",
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
    "date_calc",
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

# Create a DataFrame with the same default value for all the specific columns
data = {header: [DEFAULT_VALUE] * len(features) for header in csv_headers}
data["feature"] = features  # Set the actual feature names
data["update_time"] = [current_time] * len(features)  # Set the current time
data["status"] = ["Initialized"] * len(features)  # Set the initial status

df = pd.DataFrame(data)

# Don't overwrite file with reinitialized features
# Save DataFrame to CSV
# df.to_csv('features/features.csv', index=False)

print(f"CSV initialized with {len(features)} features.")
