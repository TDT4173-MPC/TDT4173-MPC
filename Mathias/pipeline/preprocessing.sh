#!/bin/bash

################################################################################
#                               Virtual environment                            #
################################################################################

# Source Conda's shell functions
source $(conda info --base)/etc/profile.d/conda.sh

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
fi

################################################################################
#                           Preprocessing pipeline                             #
################################################################################

echo "Starting preprocessing..."

# Import data and resample to 1 hour intervals
python3 ./Mathias/pipeline/import.py

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

python3 ./Mathias/pipeline/keep_columns.py "$COLUMNS_TO_KEEP"

echo "Preprocessing completed."

