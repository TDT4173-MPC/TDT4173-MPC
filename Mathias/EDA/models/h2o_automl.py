import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import os

# Function to process each dataset
def process_dataset(file_path, save_dir, max_runtime_secs):
    # Load data
    df = pd.read_parquet(file_path)
    
    # Split df into training and validation sets
    train_df, test_df = train_test_split(df, test_size=0.01, shuffle=False)

    train = h2o.H2OFrame(train_df)
    test = h2o.H2OFrame(test_df)

    # Identify predictors and response
    x = train.columns
    y = "pv_measurement"
    x.remove(y)

    # Run AutoML for a specified max runtime
    aml = H2OAutoML(max_models=50, seed=1, max_runtime_secs=max_runtime_secs)
    aml.train(x=x, y=y, training_frame=train)

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)

    # Number of models you want to save
    num_models_to_save = 10

    # Extract the model ids of the top models
    top_models = []
    for i in range(min(num_models_to_save, lb.nrows)):
        model_id = lb[i, 0]
        top_models.append(h2o.get_model(model_id))

    # Define the directory where you want to save the models
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create save directory if it does not exist

    # Save the models
    for model in top_models:
        h2o.save_model(model=model, path=save_dir, force=True)  # force=True to overwrite

    return aml.leader

def main():
    # Start the H2O cluster (locally)
    h2o.init(max_mem_size="10G")

    # The datasets
    datasets = {
        "A": "preprocessing/data/obs_A.parquet",
        "B": "preprocessing/data/obs_B.parquet",
        "C": "preprocessing/data/obs_C.parquet"
    }

    # Max runtime for each dataset in seconds (3 hours)
    max_runtime_secs = 3 * 3600

    leaders = {}
    # Process each dataset
    for set_name, file_path in datasets.items():
        print(f"Processing Set {set_name}")
        save_directory = f"Mathias/models/h2o_models/{set_name}"
        leaders[set_name] = process_dataset(file_path, save_directory, max_runtime_secs)

    # You can access each leader model from the leaders dictionary
    print(leaders)

    # Optional: If you wish to shut down the H2O cluster after processing
    h2o.cluster().shutdown()

if __name__ == "__main__":
    main()
