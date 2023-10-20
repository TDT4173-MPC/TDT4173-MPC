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
"keep_columns.py" \
"add_time_features.py" \
"remove_constants.py"
# "add_fourier_terms.py"
#"add_calc_time.py"
)

# Define scripts to run on training data files
SCRIPTS=(
"handle_nan.py" \
#"remove_outliers.py" \
"normalize.py" \
"remove_outliers.py" \
"normalize_pressure.py"
)

# Remove columns that are not needed
COLUMNS_TO_KEEP="\
pv_measurement \
date_forecast \
diffuse_rad:W \
diffuse_rad_1h:J \
direct_rad:W \
direct_rad_1h:J \
effective_cloud_cover:p \
fresh_snow_24h:cm \
is_day:idx \
is_in_shadow:idx \
sun_elevation:d \
super_cooled_liquid_water:kgm2 \
t_1000hPa:K \
total_cloud_cover:p \
visibility:m \
wind_speed_10m:ms \
wind_speed_u_10m:ms \
wind_speed_v_10m:ms"


COLUMNS_LEFT="\
elevation:m \
date_calc \
msl_pressure:hPa \
precip_5min:mm \
precip_type_5min:idx \
pressure_100m:hPa \
pressure_50m:hPa \
rain_water:kgm2 \
relative_humidity_1000hPa:p \
sfc_pressure:hPa \
snow_depth:cm \
snow_melt_10min:mm \
snow_water:kgm2 \
sun_azimuth:d \
absolute_humidity_2m:gm3 \
air_density_2m:kgm3 \
ceiling_height_agl:m \
clear_sky_energy_1h:J \
clear_sky_rad:W \
cloud_base_agl:m \
dew_point_2m:K \
prob_rime:p \
dew_or_rime:idx \
fresh_snow_3h:cm \
snow_density:kgm3 \
fresh_snow_6h:cm \
fresh_snow_12h:cm \
fresh_snow_1h:cm \
wind_speed_w_1000hPa:ms \
snow_drift:idx"


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

