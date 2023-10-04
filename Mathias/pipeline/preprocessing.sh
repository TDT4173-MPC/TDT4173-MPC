#!/bin/bash

LOG_FILE="./Analysis/Mathias/pipeline/preprocessing.log"
SCRIPT_PATH="./Analysis/Mathias/pipeline"
DATA_PATH="./Analysis/Mathias/pipeline/data"

# Clear the log file
> $LOG_FILE



################################################################################
#                           Preprocessing pipeline                             #
################################################################################

echo -e "\nStarting preprocessing..." | tee -a $LOG_FILE

# Import data and resample to 1 hour intervals
FILES=$(python3 $SCRIPT_PATH/import.py)
if [ $? -ne 0 ]; then
    echo "Error during data import. Exiting." | tee -a $LOG_FILE
    exit 1
fi
echo -e "\nFiles to process:\n$FILES\n" | tee -a $LOG_FILE

# Remove columns that are not needed
COLUMNS_TO_KEEP="\
pv_measurement \
absolute_humidity_2m:gm3 \
air_density_2m:kgm3 \
clear_sky_energy_1h:J \
clear_sky_rad:W \
dew_point_2m:K \
diffuse_rad:W \
diffuse_rad_1h:J \
direct_rad:W \
direct_rad_1h:J \
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
super_cooled_liquid_water:kgm2 \
t_1000hPa:K \
total_cloud_cover:p \
visibility:m \
wind_speed_10m:ms \
wind_speed_u_10m:ms \
wind_speed_v_10m:ms"


echo -e "Keeping columns:\n$COLUMNS_TO_KEEP\n" | tee -a $LOG_FILE

for file in $FILES; do
    python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error processing columns for $file. Exiting." | tee -a $LOG_FILE
        exit 1
    fi
done

# Removing outliers
# echo -e "\nRemoving outliers..." | tee -a $LOG_FILE
# for file in $FILES; do
#     python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
#     if [ ${PIPESTATUS[0]} -ne 0 ]; then
#         echo "Error removing outliers for $file. Exiting." | tee -a $LOG_FILE
#         exit 1
#     fi
# done

# # Interpolate missing values
# echo -e "\nInterpolating missing values..." | tee -a $LOG_FILE
# for file in $FILES; do
#     python3 $SCRIPT_PATH/interpolate.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
#     if [ ${PIPESTATUS[0]} -ne 0 ]; then
#         echo "Error interpolating for $file. Exiting." | tee -a $LOG_FILE
#         exit 1
#     fi
# done

# Handle NaNs
echo -e "\nHandling NaNs..." | tee -a $LOG_FILE
for file in $FILES; do
    python3 $SCRIPT_PATH/handle_nan.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error handling NaNs for $file. Exiting." | tee -a $LOG_FILE
        exit 1
    fi
done

# # Add features
# echo -e "\nAdding features..." | tee -a $LOG_FILE
# for file in $FILES; do
#     python3 $SCRIPT_PATH/add_features.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
#     if [ ${PIPESTATUS[0]} -ne 0 ]; then
#         echo "Error adding features for $file. Exiting." | tee -a $LOG_FILE
#         exit 1
#     fi
# done

# Normalize data
# echo -e "\nNormalizing data..." | tee -a $LOG_FILE
# for file in $FILES; do
#     python3 $SCRIPT_PATH/normalize.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
#     if [ ${PIPESTATUS[0]} -ne 0 ]; then
#         echo "Error normalizing $file. Exiting." | tee -a $LOG_FILE
#         exit 1
#     fi
# done

echo -e "\nPreprocessing completed\n" | tee -a $LOG_FILE

# Ploting
# TARGET="pv_measurement"
# FEATURE="sfc_pressure:hPa"
# FILE="$DATA_PATH/obs_A.csv"
# echo -e "\nPlotting..." | tee -a $LOG_FILE
# python3 $SCRIPT_PATH/plot.py $FILE $FEATURE $TARGET 2>&1 | tee -a $LOG_FILE
# if [ ${PIPESTATUS[0]} -ne 0 ]; then
#     echo "Error plotting for $file. Exiting." | tee -a $LOG_FILE
#     exit 1
# fi



