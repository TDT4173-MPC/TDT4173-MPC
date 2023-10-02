#!/bin/bash

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
    echo "Activating conda environment: $ENV_NAME"
    conda activate $ENV_NAME
else
    echo "Creating conda environment: $ENV_NAME from $ENV_PATH"
    conda env create -f $ENV_PATH
    conda activate $ENV_NAME
fi

# Ensure the environment was activated successfully
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Error activating environment $ENV_NAME. Exiting."
    exit 1
else
    echo "Successfully activated conda environment: $ENV_NAME"
fi

################################################################################
#                           Preprocessing pipeline                             #
################################################################################

echo "Starting preprocessing..."

SCRIPT_PATH="./Mathias/pipeline"
DATA_PATH="./Mathias/pipeline/data"

# Import data and resample to 1 hour intervals
python3 $SCRIPT_PATH/import.py

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

echo "Keeping columns: $COLUMNS_TO_KEEP"

python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/x_obs_A.csv
python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/x_est_A.csv
python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/x_obs_B.csv
python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/x_est_B.csv
python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/x_obs_C.csv
python3 $SCRIPT_PATH/keep_columns.py "$COLUMNS_TO_KEEP" $DATA_PATH/x_est_C.csv

# Remove outliers and NaNs
python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/x_obs_A.csv
python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/x_est_A.csv
python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/x_obs_B.csv
python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/x_est_B.csv
python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/x_obs_C.csv
python3 $SCRIPT_PATH/remove_outliers.py $DATA_PATH/x_est_C.csv

# Normalize data
python3 $SCRIPT_PATH/normalize.py $DATA_PATH/x_obs_A.csv
python3 $SCRIPT_PATH/normalize.py $DATA_PATH/x_est_A.csv
python3 $SCRIPT_PATH/normalize.py $DATA_PATH/x_obs_B.csv
python3 $SCRIPT_PATH/normalize.py $DATA_PATH/x_est_B.csv
python3 $SCRIPT_PATH/normalize.py $DATA_PATH/x_obs_C.csv
python3 $SCRIPT_PATH/normalize.py $DATA_PATH/x_est_C.csv

echo "Preprocessing completed."

