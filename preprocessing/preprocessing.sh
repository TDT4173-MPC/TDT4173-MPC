#!/bin/bash

LOG_FILE="preprocessing/preprocessing.log"
SCRIPT_PATH="preprocessing/scripts"
DATA_PATH="preprocessing/Data"

# Clear the log file
> $LOG_FILE

################################################################################
#                           Preprocessing pipeline                             #
################################################################################

# Define scripts to run on training data files
SCRIPTS=(
"handle_nan.py" \
"normalize_pressure.py"
)

# Define scripts to run on all files including test files
SCRIPTS_ALL=(
"keep_columns.py" \
"add_time_features.py" \
"add_calc_time.py"
)

# Remove columns that are not needed
COLUMNS_TO_KEEP="\
pv_measurement \
date_forecast \
time \
absolute_humidity_2m:gm3 \
air_density_2m:kgm3 \
clear_sky_rad:W \
dew_point_2m:K \
diffuse_rad:W \
direct_rad:W \
effective_cloud_cover:p \
elevation:m \
is_day:idx \
is_in_shadow:idx \
msl_pressure:hPa \
pressure_100m:hPa \
pressure_50m:hPa \
relative_humidity_1000hPa:p \
sfc_pressure:hPa \
snow_water:kgm2 \
sun_azimuth:d \
sun_elevation:d \
t_1000hPa:K \
total_cloud_cover:p \
visibility:m \
wind_speed_10m:ms \
wind_speed_u_10m:ms \
wind_speed_v_10m:ms \
clear_sky_energy_1h:J \
diffuse_rad_1h:J \
direct_rad_1h:J \
date_calc"

COLUMNS_LEFT="\
super_cooled_liquid_water:kgm2"


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

