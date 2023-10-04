################################################################################
#                               Virtual environment                            #
################################################################################

# Run file from Analysis directory

# Source Conda's shell functions
source ~/anaconda3/bin/activate

# Name of the conda environment
ENV_NAME="TDT4173-MPC"
ENV_PATH="./venv/TDT4173-MPC.yml"
LOG_FILE="./venv/venv_setup.log"

# Check if the conda environment already exists
if conda info --envs | grep -q $ENV_NAME; then
    echo -e "\nConda environment $ENV_NAME already exists." | tee -a $LOG_FILE
else
    echo -e "\nCreating conda environment: $ENV_NAME from $ENV_PATH" | tee -a $LOG_FILE
    conda env create -f $ENV_PATH
fi

# Update the environment to ensure all dependencies in the .yml file are installed
echo -e "\nUpdating conda environment: $ENV_NAME based on $ENV_PATH" | tee -a $LOG_FILE
conda env update -f $ENV_PATH --prune

# Print the activation command
echo -e "\nRun the following command to activate the environment:\n"
echo -e "conda activate $ENV_NAME\n"
