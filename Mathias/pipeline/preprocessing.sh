#!/bin/bash

LOG_FILE="./Mathias/pipeline/preprocessing.log"

# Clear the log file
> $LOG_FILE


################################################################################
#                               Virtual environment                            #
################################################################################

# Source Conda's shell functions
source ~/anaconda3/bin/activate

# Name of the conda environment
ENV_NAME="TDT4173-MPC"
ENV_PATH="Analysis/TDT4173-MPC.yml"

# Check if the conda environment already exists
if conda info --envs | grep -q $ENV_NAME; then
    echo -e "\nActivating conda environment: $ENV_NAME" | tee -a $LOG_FILE
    conda activate $ENV_NAME
else
    echo -e "\nCreating conda environment: $ENV_NAME from $ENV_PATH" | tee -a $LOG_FILE
    conda env create -f $ENV_PATH
    conda activate $ENV_NAME
fi

# Ensure the environment was activated successfully
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Error activating environment $ENV_NAME. Exiting." | tee -a $LOG_FILE
    exit 1
else
    echo "Successfully activated conda environment: $ENV_NAME" | tee -a $LOG_FILE
fi


################################################################################
#                           Preprocessing pipeline                             #
################################################################################

echo -e "\nStarting preprocessing..." | tee -a $LOG_FILE

SCRIPT_PATH="./Mathias/pipeline"
DATA_PATH="./Mathias/pipeline/data"

# Import data and resample to 1 hour intervals
FILES=$(python3 $SCRIPT_PATH/import.py)
if [ $? -ne 0 ]; then
    echo "Error during data import. Exiting." | tee -a $LOG_FILE
    exit 1
fi
echo -e "\nFiles to process:\n$FILES\n" | tee -a $LOG_FILE

# Remove columns that are not needed
COLUMNS_TO_KEEP="\
time clear_sky_rad:W \
clear_sky_energy_1h:J \
sun_elevation:d \
is_day:idx \
direct_rad_1h:J \
pv_measurement \
diffuse_rad_1h:J \
pressure_100m:hPa"

echo -e "Keeping columns:\n$COLUMNS_TO_KEEP\n" | tee -a $LOG_FILE

for file in $FILES; do
    python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error processing columns for $file. Exiting." | tee -a $LOG_FILE
        exit 1
    fi
done

# Removing outliers and NaN rows
echo -e "\nRemoving outliers and NaN rows..." | tee -a $LOG_FILE
for file in $FILES; do
    python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error removing outliers for $file. Exiting." | tee -a $LOG_FILE
        exit 1
    fi
done

# Normalize data
echo -e "\nNormalizing data..." | tee -a $LOG_FILE
for file in $FILES; do
    python3 $SCRIPT_PATH/normalize.py $DATA_PATH/$file 2>&1 | tee -a $LOG_FILE
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error normalizing $file. Exiting." | tee -a $LOG_FILE
        exit 1
    fi
done

echo -e "\nPreprocessing completed\n" | tee -a $LOG_FILE




