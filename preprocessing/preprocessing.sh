#!/bin/bash

LOG_FILE="preprocessing/preprocessing.log"
SCRIPT_PATH="preprocessing/scripts"
DATA_PATH="preprocessing/data"

# Clear the log file
> $LOG_FILE

################################################################################
#                           Preprocessing pipeline                             #
################################################################################

# Define scripts to run on all files including test files
SCRIPTS_ALL=(
# "handle_nan.py" \
"feature_testing.py" \
"add_time_features.py" \
"add_fourier_features.py" \
"add_rate_of_change.py" \
"add_obs_est_feature.py"
"remove_constants.py" \
"add_lagged_features_dict.py" \
"add_rolling_window_features.py" \
"keep_columns.py" \
"impute_nans.py" \
"power_transform.py" \
# "normalize.py" \
# "add_cosines.py"
# "add_fourier_terms.py"
#"add_calc_time.py"
)

# Define scripts to run on training data files
SCRIPTS=(
# "remove_outliers.py"
# "normalize_pressure.py"
)

# Remove columns that are not needed
COLUMNS_TO_KEEP="
pv_measurement \
snow_accumulation \
total_radiation \
sfc_pressure:hPa \
month \
year \
date_forecast_fft_amplitude \
date_forecast_fft_phase \
sun_elevation:d_fft_amplitude \
sun_elevation:d_fft_phase \
t_1000hPa:K_rate_of_change \
clear_sky_rad:W_rate_of_change \
direct_rad:W_rate_of_change \
effective_cloud_cover:p_rate_of_change \
total_cloud_cover:p_rate_of_change \
observed \
sun_azimuth:d_lag_7 \
msl_pressure:hPa_lag_3 \
sfc_pressure:hPa_lag_8 \
t_1000hPa:K_lag_4 \
dew_or_rime:idx_lag_11 \
relative_humidity_1000hPa:p_lag_-3 \
temp_dewpoint_diff_lag_-4 \
dew_point_2m:K_lag_19 \
visibility:m_lag_-2 \
t_1000hPa:K_rolling_avg_24 \
msl_pressure:hPa_rolling_avg_24 \
absolute_humidity_2m:gm3_rolling_avg_24 \
total_cloud_cover:p_rolling_avg_6 \
sun_elevation:d_rolling_avg_6 \
total_radiation_rolling_avg_3 \

diffuse_rad:W \
diffuse_rad_1h:J \
direct_rad:W \
direct_rad_1h:J \
clear_sky_rad:W \
sun_elevation:d \
effective_cloud_cover:p \
clear_sky_energy_1h:J"

# pv_measurement \
# date_forecast \
# absolute_humidity_2m:gm3 \
# super_cooled_liquid_water:kgm2 \
# t_1000hPa:K \
# total_cloud_cover:p \
# air_density_2m:kgm3 \
# visibility:m \
# msl_pressure:hPa \
# dew_point_2m:K \
# relative_humidity_1000hPa:p \
# snow_water:kgm2 \
# snow_accumulation \
# wind_vector_magnitude \
# average_wind_speed \
# pressure_gradient \
# temp_dewpoint_diff \

# \
# date_calc \
# precip_5min:mm \
# is_day:idx \
# is_in_shadow:idx \
# precip_type_5min:idx \
# pressure_100m:hPa \
# pressure_50m:hPa \
# rain_water:kgm2 \
# sfc_pressure:hPa \
# snow_depth:cm \
# snow_melt_10min:mm \
# sun_azimuth:d \
# prob_rime:p \
# dew_or_rime:idx" \


COLUMNS_LEFT="\
wind_speed_w_1000hPa:ms \
wind_speed_u_10m:ms \
wind_speed_v_10m:ms \
wind_speed_10m:ms \
fresh_snow_24h:cm \
fresh_snow_12h:cm \
fresh_snow_6h:cm \
fresh_snow_3h:cm \
fresh_snow_1h:cm \
elevation:m" \

# Columns that messes up test data
# ceiling_height_agl:m
# cloud_base_agl:m
# snow_density:kgm3
# snow_drift:idx


# Print info to log file
echo -e "\nStarting preprocessing..." | tee -a $LOG_FILE
echo -e "\nKeeping columns:\n$COLUMNS_TO_KEEP\n" | tee -a $LOG_FILE


# Import data and resample to 1 hour intervals
FILES=$(python3 $SCRIPT_PATH/import.py)
if [ $? -ne 0 ]; then
    echo "Error during data import. Exiting." | tee -a $LOG_FILE
    exit 1
fi
echo -e "\nFiles to process:\n$FILES\n" | tee -a $LOG_FILE

# Define a function to process files
process_files() {
    local script_name="$1"
    
    echo -e "\nRunning $script_name..." | tee -a $LOG_FILE
    for file in $FILES; do
        if [ "$script_name" == "keep_columns.py" ]; then
            python3 $SCRIPT_PATH/$script_name "$COLUMNS_TO_KEEP" $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
        else
            python3 $SCRIPT_PATH/$script_name $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
        fi

        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Error with $script_name for $file. Exiting." | tee -a $LOG_FILE
            exit 1
        fi
    done
}

# Scripts to run on all files including test files
for script in "${SCRIPTS_ALL[@]}"; do
    process_files "$script"
done

# Remove test files from FILES variable
FILES=$(echo "$FILES" | grep -v "test_")

# Loop through the remaining scripts and process files
for script in "${SCRIPTS[@]}"; do
    process_files "$script"
done

echo -e "\nPreprocessing completed\n" | tee -a $LOG_FILE

