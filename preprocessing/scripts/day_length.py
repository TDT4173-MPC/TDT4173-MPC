import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun, SunDirection
import pytz  
import sys

def add_daylight_features(df, latitude, longitude, timezone):
    """
    Adds 'day_length_hours', 'time_to_noon', and 'time_from_noon' columns to the dataframe.
    Assumes 'date_forecast' column exists and is in datetime format.
    """
    city = LocationInfo(latitude=latitude, longitude=longitude, timezone=timezone)
    tz = pytz.timezone(timezone)  # Create a timezone object

    # Initialize new columns with default values
    df['day_length_hours'] = np.nan
    df['time_to_noon'] = np.nan
    df['time_from_noon'] = np.nan

    for index, row in df.iterrows():
        # Localize 'date_forecast' to the specified timezone or make it offset-aware
        localized_forecast_time = tz.localize(row['date_forecast'].to_pydatetime())

        try:
            s = sun(city.observer, date=localized_forecast_time.date(), tzinfo=tz)
        except ValueError as e:  # Catch the ValueError if dusk or dawn times cannot be calculated
            print(f"Warning: {e} for date {localized_forecast_time.date()}.")
            # Handle the extreme condition by setting a default value or skipping the calculation
            continue  # Skip this row or set your own logic for default values here

        sunrise = s['sunrise']
        sunset = s['sunset']
        solar_noon = s['noon']

        # Calculate day length in hours
        day_length = sunset - sunrise
        df.at[index, 'day_length_hours'] = day_length.total_seconds() / 3600

        # Calculate time to/from solar noon in hours
        if localized_forecast_time <= solar_noon:
            df.at[index, 'time_to_noon'] = (solar_noon - localized_forecast_time).total_seconds() / 3600
            df.at[index, 'time_from_noon'] = 0
        else:
            df.at[index, 'time_from_noon'] = (localized_forecast_time - solar_noon).total_seconds() / 3600
            df.at[index, 'time_to_noon'] = 0

    # df['day_length_radiation_interaction'] = df['day_length_hours'] * df['total_radiation']
    df['alternative_cloud_day_interaction'] = df['total_cloud_cover:p'] * np.sin(np.pi * (1 - abs(df['time_to_noon'] - df['time_from_noon']) / df['day_length_hours'])) * df['is_day:idx']

  

    return df

def main(input_file, latitude, longitude, timezone):
    # Read the data
    df = pd.read_parquet(input_file)

    # Ensure 'date_forecast' is a datetime column
    df['date_forecast'] = pd.to_datetime(df['date_forecast'])

    # Add daylight features
    modified_df = add_daylight_features(df, latitude, longitude, timezone)

    # Save the modified data back to the same path
    modified_df.to_parquet(input_file, index=False)
    print(f"Daylight features added and file saved to {input_file}.")

if __name__ == "__main__":
    input_file_path = sys.argv[1]
    norway_latitude = 59.91  # Oslo's latitude as an example
    norway_longitude = 10.75  # Oslo's longitude as an example
    norway_timezone = 'Europe/Oslo'  # Oslo's timezone as an example
    main(input_file_path, norway_latitude, norway_longitude, norway_timezone)
