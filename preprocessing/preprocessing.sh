#!/bin/bash

LOG_FILE="preprocessing/preprocessing.log"
SCRIPT_PATH="preprocessing/scripts"
DATA_PATH="preprocessing/data"
RUN_FEATURE_TESTING=true

# Clear the log file
> $LOG_FILE

################################################################################
#                           Preprocessing pipeline                             #
################################################################################

# Define scripts to run on all files including test files
SCRIPTS_ALL=(
"feature_testing.py" \
"add_time_features.py" \
# "add_fourier_features.py" \
# "add_rate_of_change.py" \
# "add_obs_est_feature.py"
"remove_constants.py" \
# "add_lagged_features_dict.py" \
# "add_rolling_window_features.py" \
"keep_columns.py" \
# "add_polynomial_features.py" \
# "add_angled_features.py" \
"handle_nan.py" \
# "pca.py" \
# "impute_nans.py" \
# "power_transform.py" \
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

COLUMNS_FILE="$1"

if [ -z "$COLUMNS_FILE" ] || [ ! -f "$COLUMNS_FILE" ]; then
    echo "You must provide a valid file containing columns to keep as the first argument."
    exit 1
fi

# Read columns to keep from the file
COLUMNS_TO_KEEP=$(cat "$COLUMNS_FILE")

# Remove columns that are not needed
# COLUMNS_TO_KEEP="
# pv_measurement \
# date_forecast \
# snow_accumulation \
# total_radiation \
# sfc_pressure:hPa \
# month \
# year \
# date_forecast_fft_amplitude \
# date_forecast_fft_phase \
# sun_elevation:d_fft_amplitude \
# sun_elevation:d_fft_phase \
# clear_sky_rad:W_rate_of_change \
# direct_rad:W_rate_of_change \
# diffuse_rad:W_rate_of_change \
# total_radiation_rate_of_change \
# effective_cloud_cover:p_rate_of_change \
# total_cloud_cover:p_rate_of_change \
# observed \
# sun_azimuth:d_lag_7 \
# sfc_pressure:hPa_lag_8 \
# t_1000hPa:K_lag_4 \
# dew_or_rime:idx_lag_11 \
# relative_humidity_1000hPa:p_lag_-3 \
# temp_dewpoint_diff_lag_-4 \
# dew_point_2m:K_lag_19 \
# visibility:m_lag_-2 \
# t_1000hPa:K_rolling_avg_24 \
# msl_pressure:hPa_rolling_avg_24 \
# absolute_humidity_2m:gm3_rolling_avg_24 \
# total_cloud_cover:p_rolling_avg_6 \
# total_radiation_rolling_avg_3 \
# diffuse_rad:W \
# diffuse_rad_1h:J \
# direct_rad:W \
# direct_rad_1h:J \
# clear_sky_rad:W \
# sun_elevation:d \
# t_1000hPa:K \
# effective_cloud_cover:p \
# clear_sky_energy_1h:J \

# sun_elevation_direct_rad_interaction \
# humidity_temp_interaction \
# temp_rad_interaction \
# total_cloud_cover:p \
# air_density_2m:kgm3 \
# msl_pressure:hPa_lag_3 \
# t_1000hPa:K_rate_of_change \
# wind_vector_magnitude \
# average_wind_speed \
# visibility:m \
# dew_point_2m:K \
# pressure_gradient \
# temp_dewpoint_diff \
# sun_elevation:d_rolling_avg_6 \
# super_cooled_liquid_water:kgm2 \
# relative_humidity_1000hPa:p \
# is_day:idx \
# is_in_shadow:idx \
# pressure_50m:hPa \
# sun_azimuth:d \
# absolute_humidity_2m:gm3" \

# date_calc \
# snow_water:kgm2 \ 
# precip_5min:mm \
# precip_type_5min:idx \
# pressure_100m:hPa \
# rain_water:kgm2 \
# snow_depth:cm \
# snow_melt_10min:mm \
# prob_rime:p \
# dew_or_rime:idx" \


# COLUMNS_LEFT="\
# wind_speed_w_1000hPa:ms \
# wind_speed_u_10m:ms \
# wind_speed_v_10m:ms \
# wind_speed_10m:ms \
# msl_pressure:hPa \
# fresh_snow_24h:cm \
# fresh_snow_12h:cm \
# fresh_snow_6h:cm \
# fresh_snow_3h:cm \
# fresh_snow_1h:cm \
# elevation:m" \

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


################################################################################
#                           Feature testing                                    #
################################################################################

if [ "$RUN_FEATURE_TESTING" = true ]; then

    # Define pairs of observations and estimates
    declare -a DATA_PAIRS=("obs_A,est_A" "obs_B,est_B" "obs_C,est_C")

    # Now, iterate over each pair and run test_features.py
    for pair in "${DATA_PAIRS[@]}"; do
        # Split the pair into two separate variables
        IFS=',' read -r obs est <<< "$pair"

        # Construct file paths using the constants defined earlier
        obs_file="$DATA_PATH/${obs}.parquet"
        est_file="$DATA_PATH/${est}.parquet"

        # Check if the files exist before running the test_features.py script
        if [[ -f "$obs_file" && -f "$est_file" ]]; then
            echo -e "\nRunning test_features.py with $obs and $est..." | tee -a $LOG_FILE
            python3 $SCRIPT_PATH/test_features.py "$obs_file" "$est_file" 2>&1 | tee -a $LOG_FILE
            # Capture the exit status of the last command
            status=$?
            if [ $status -ne 0 ]; then
                echo "Error during feature testing with $obs and $est (Exit Status: $status). Exiting." | tee -a $LOG_FILE
                exit $status
            fi
        else
            echo "Required files for test_features.py with $obs and $est not found. Skipping this pair." | tee -a $LOG_FILE
            # Do not exit the script if files are not found, just skip to the next pair
        fi
    done

    echo -e "\nFeature testing for all pairs completed\n" | tee -a $LOG_FILE

fi
